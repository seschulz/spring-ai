/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.bedrock.converse.api;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import software.amazon.awssdk.services.bedrockruntime.model.*;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import software.amazon.awssdk.services.bedrockruntime.model.conversestreamoutput.*;

import javax.management.relation.Role;

import static org.assertj.core.api.Assertions.assertThat;


class ConverseApiUtilsTest {

	@Test
	void bufferedAssistantMessageContentIsUsedForToolCall() {
	
		AtomicInteger index = new AtomicInteger(0);
		List<ConverseStreamOutput> events = new ArrayList<>();
		String assistantMessageText = "Let me get the weather forecast for you";
		String toolUseId = "toolUseId1";
		String toolUseName = "weather";
		String toolUseArguments ="{\"latitude\":\"34.0522\", \"longitude\":\"-118.2427\"}";

		// Arrange:
		// Assistant response to announce tool call
		events.add(DefaultMessageStart.builder().role(ConversationRole.ASSISTANT).build());
		events.add(DefaultContentBlockDelta.builder().contentBlockIndex(index.get())
				.delta(ContentBlockDelta.builder().text(assistantMessageText).build()).build());
		events.add(DefaultContentBlockStop.builder().contentBlockIndex(index.get()).build());

		// Tool use
		events.add(DefaultContentBlockStart.builder()
				.contentBlockIndex(index.addAndGet(1))
				.start(ContentBlockStart.builder()
						.toolUse(ToolUseBlockStart.builder().toolUseId(toolUseId).name(toolUseName).build())
						.build())
				.build());
		events.add(DefaultContentBlockDelta.builder().contentBlockIndex(index.get())
				.delta(ContentBlockDelta.builder()
						.toolUse(ToolUseBlockDelta.builder().input(toolUseArguments).build())
						.build())
				.build());
		events.add(DefaultContentBlockStop.builder().contentBlockIndex(index.get()).build());
		events.add(DefaultMessageStop.builder().stopReason(StopReason.TOOL_USE).build());
		
		// Metadata
		events.add(DefaultMetadata.builder()
				.usage(TokenUsage.builder().inputTokens(10).outputTokens(5).totalTokens(15).build())
						.metrics(ConverseStreamMetrics.builder().latencyMs(1234L).build())
				.build());
		Flux<ConverseStreamOutput> eventsFlux = Flux.fromIterable(events);

		// Act
		List<ChatResponse> responses = ConverseApiUtils.toChatResponse(eventsFlux, null).collectList().block();

		// Assert: find assistantMessageText in tool call
		assertThat(responses).isNotNull();
		Generation toolCallGen = responses.stream()
			.flatMap(r -> r.getResults().stream())
			.filter(g -> !g.getOutput().getToolCalls().isEmpty())
			.findFirst()
			.orElseThrow(() -> new AssertionError("No tool call generation emitted"));

		AssistantMessage assistantMsg = (AssistantMessage) toolCallGen.getOutput();
		assertThat(assistantMsg.getText()).isEqualTo(assistantMessageText);
		assertThat(assistantMsg.getToolCalls()).hasSize(1);
		AssistantMessage.ToolCall tc = assistantMsg.getToolCalls().get(0);
		assertThat(tc.id()).isEqualTo(toolUseId);
		assertThat(tc.name()).isEqualTo(toolUseName);
		assertThat(tc.arguments()).isEqualTo(toolUseArguments);
	}

}
