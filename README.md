### The answer of the project is in Answer_XIA.ipynb

The result of the code is not as good as expected, primarily because the Gen-AI model I chose was GPT-2 Small, due to local GPU and memory constraints. I initially tried using CodeLlama and GPT2-medium, but had to abandon it for the same resource limitations. GPT-2, being a much smaller model, was more compatible with my hardware, although its performance on generative tasks is limited.

My approach involved using CLIP, a model trained to associate images with their corresponding textual descriptions via a contrastive pre-training objective. During training, CLIP learns to bring the embeddings of an image and its corresponding CadQuery code closer together, while pushing apart embeddings of unrelated pairs.

In theory, by leveraging CLIP to extract image features aligned with their corresponding CadQuery code, we aim to build a joint representation that connects the visual CAD images with the procedural code generation process, improving the model’s understanding and output quality.

Due to both time constraints and hardware limitations, I was only able to use 0.2% of the full dataset for training. Using more data would have exceeded the time or memory available on my machine. As a result, the performance of the trained model is poor — the Valid Syntax Rate (VSR) is 0, and the training loss remains unstable and fluctuating.

To improve this, I implemented a Retrieval-Augmented Generation (RAG) strategy. For each training sample, the top-3 most similar image features (computed by CLIP) are retrieved from a 1% subset of the dataset, and their corresponding CadQuery codes are appended to the prompt. The intention is to enrich the context and provide additional guidance to the model.

Although the VSR remains at 0 — indicating that the model still fails to generate syntactically valid CadQuery code — the training loss becomes more stable and shows signs of convergence. This suggests that RAG introduces some optimization benefits, especially in learning image–code associations. However, the main bottleneck remains the limited capacity of GPT-2 Small, which lacks the parameters and scale needed to effectively model this complex task, especially given the limited training steps and dataset size.
