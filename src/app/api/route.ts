import { HfInference } from "@huggingface/inference";
import { HuggingFaceStream, StreamingTextResponse } from "ai";

const Hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

export async function POST(req: Request) {
  try {
    const { prompts } = await req.json();
    const response = await Hf.textGenerationStream({
      model: "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
      inputs: `<|prompter|>${prompts}<|endoftext|><|assistant|>`,
      parameters: {
        max_new_tokens: 200,
        // @ts-ignore (this is a valid parameter specifically in OpenAssistant models)
        typical_p: 0.2,
        repetition_penalty: 1,
        truncate: 1000,
        return_full_text: false,
      },
    });
    const stream = HuggingFaceStream(response);
    return new StreamingTextResponse(stream);
  } catch (err) {
    console.error(err);
  }
}
