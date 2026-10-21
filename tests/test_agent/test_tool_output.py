"""A tool result that carries images beside its text."""

from __future__ import annotations

from msgspec import UNSET

from plyngent.agent import ToolOutput, ToolRegistry, tool
from plyngent.agent.budget import IMAGE_CHARS, estimate_message_chars
from plyngent.agent.loop import _run_one_tool
from plyngent.lmproto.openai_compatible.model import (
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatImagePart,
    ChatImageUrl,
    ChatTextPart,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.tools.context import InstanceState


def _image() -> ChatImagePart:
    return ChatImagePart(image_url=ChatImageUrl(url="data:image/png;base64,AAAA", detail="low"))


def _registry() -> ToolRegistry:
    @tool(register=False)
    def shot() -> ToolOutput:
        """Pretend to take a screenshot."""
        return ToolOutput(text="image: shot.png (2 KB)", images=[_image()])

    @tool(register=False)
    def plain() -> str:
        return "just text"

    return ToolRegistry([shot, plain])


def _call(name: str) -> AssistantFunctionToolCall:
    return AssistantFunctionToolCall(id="call-1", function=AssistantFunctionTool(name=name, arguments="{}"))


async def test_execute_keeps_the_text_only_contract() -> None:
    registry = _registry()

    assert await registry.execute("shot", "{}") == "image: shot.png (2 KB)"
    assert await registry.execute("plain", "{}") == "just text"


async def test_execute_output_returns_the_images_beside_the_text() -> None:
    registry = _registry()
    output = await registry.execute_output("shot", "{}")

    assert output.text == "image: shot.png (2 KB)"
    assert output.has_images
    assert output.images == [_image()]


async def test_execute_output_wraps_a_plain_string_result() -> None:
    output = await _registry().execute_output("plain", "{}")

    assert output.text == "just text"
    assert not output.has_images


async def test_an_unknown_tool_is_error_text_without_images() -> None:
    output = await _registry().execute_output("missing", "{}")

    assert "unknown tool" in output.text
    assert not output.has_images


async def test_the_loop_keeps_tool_images_on_the_tool_message() -> None:
    message, err = await _run_one_tool(_registry(), _call("shot"), max_result_chars=32_000)

    assert err is None
    assert isinstance(message, ToolChatMessage)
    assert message.content == "image: shot.png (2 KB)"
    assert message.images == [_image()]


async def test_the_loop_leaves_the_images_field_unset_for_text_tools() -> None:
    message, err = await _run_one_tool(_registry(), _call("plain"), max_result_chars=32_000)

    assert err is None
    assert message.images is UNSET


async def test_a_truncated_tool_result_keeps_its_images() -> None:
    message, _ = await _run_one_tool(_registry(), _call("shot"), max_result_chars=1)

    assert "[Truncated" in message.content
    assert message.images == [_image()]


def test_the_char_estimate_counts_images() -> None:
    assert estimate_message_chars(UserChatMessage(content="look:")) == len("look:")
    assert estimate_message_chars(UserChatMessage(content=[ChatTextPart(text="look:"), _image()])) == (
        len("look:") + IMAGE_CHARS
    )
    assert estimate_message_chars(ToolChatMessage(content="shot", tool_call_id="call-1", images=[_image()])) == (
        len("shot") + len("call-1") + IMAGE_CHARS
    )


def _registry_for(instance: InstanceState) -> ToolRegistry:
    @tool(register=False)
    def shot() -> ToolOutput:
        """Pretend to take a screenshot."""
        return ToolOutput(text="image: shot.png (2 KB)", images=[_image()])

    return ToolRegistry([shot], instance_state=instance)


async def test_a_text_only_model_never_receives_a_plugin_tool_image() -> None:
    """The registry is the safety net for tools that do not check ``image_in``."""
    instance = InstanceState()
    instance.workspace.image_in = False
    instance.workspace.image_model = ("ds", "text-only")

    output = await _registry_for(instance).execute_output("shot", "{}")

    assert output.text.startswith("image: shot.png (2 KB)")
    assert "image dropped" in output.text
    assert "image_in = true" in output.text
    assert not output.has_images


async def test_a_vision_model_keeps_a_plugin_tool_image() -> None:
    instance = InstanceState()
    instance.workspace.image_in = True

    output = await _registry_for(instance).execute_output("shot", "{}")

    assert output.text == "image: shot.png (2 KB)"
    assert output.images == [_image()]
