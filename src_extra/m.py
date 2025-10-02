from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-5-2025-08-07",
    input="Explain diffusion vs. osmosis in simple terms.",
    reasoning={"summary": "auto", "effort": "medium"}
)

# Final answer text
output_text = getattr(resp, "output_text", "") or ""

# Reasoning summary text
reasoning_text = ""
reasoning_item = next((it for it in resp.output or [] if it.type == "reasoning"), None)
if reasoning_item and getattr(reasoning_item, "summary", None):
    reasoning_text = "\n".join(s.text for s in reasoning_item.summary)

print("=== Reasoning Summary ===\n", reasoning_text)
print("\n=== Output ===\n", output_text)
