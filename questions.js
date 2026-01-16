const questions = [
  {
    id: 1,
    question: "What is an LLM?",
    options: {
      A: "Store documents",
      B: "Predict the next token in a sequence",
      C: "Search the internet",
      D: "Reason like a human"
    },
    correctAnswer: "B"
  },
  {
    id: 2,
    question: "LLMs are typically trained using:",
    options: {
      A: "Reinforcement learning only",
      B: "Supervised classification",
      C: "Self-supervised learning on large text corpora",
      D: "Rule-based systems"
    },
    correctAnswer: "C"
  },
  {
    id: 3,
    question: "Why are LLMs called “large”?",
    options: {
      A: "They use large datasets only",
      B: "They have many parameters (billions)",
      C: "They generate long answers",
      D: "They consume more memory at runtime only"
    },
    correctAnswer: "B"
  },
  {
    id: 4,
    question: "Which task is an LLM inherently good at?",
    options: {
      A: "Exact arithmetic",
      B: "Image segmentation",
      C: "Natural language generation",
      D: "Database indexing"
    },
    correctAnswer: "C"
  },
  {
    id: 5,
    question: "LLMs fundamentally learn:",
    options: {
      A: "Facts",
      B: "Reasoning rules",
      C: "Statistical patterns in language",
      D: "Ground truth knowledge"
    },
    correctAnswer: "C"
  },
  {
    id: 6,
    question: "A token is best described as:",
    options: {
      A: "A word",
      B: "A character",
      C: "A sub-word or chunk of text",
      D: "A sentence"
    },
    correctAnswer: "C"
  },
  {
    id: 7,
    question: "Why do tokens matter?",
    options: {
      A: "They affect only cost",
      B: "They define how text is processed internally",
      C: "They control hallucinations",
      D: "They improve accuracy directly"
    },
    correctAnswer: "B"
  },
  {
    id: 8,
    question: "Context window refers to:",
    options: {
      A: "Time limit for response",
      B: "Number of users supported",
      C: "Maximum tokens model can consider at once",
      D: "Memory stored permanently"
    },
    correctAnswer: "C"
  },
  {
    id: 9,
    question: "If conversation exceeds context window:",
    options: {
      A: "Model crashes",
      B: "Oldest tokens are dropped or ignored",
      C: "Accuracy improves",
      D: "Output becomes deterministic"
    },
    correctAnswer: "B"
  },
  {
    id: 10,
    question: "Larger context windows are especially useful for:",
    options: {
      A: "Short Q&A",
      B: "Chatbots with memory",
      C: "Long documents & RAG use cases",
      D: "Image generation"
    },
    correctAnswer: "C"
  },
  {
    id: 11,
    question: "Prompt engineering means:",
    options: {
      A: "Training the model",
      B: "Designing effective input prompts",
      C: "Tuning hyperparameters",
      D: "Changing model weights"
    },
    correctAnswer: "B"
  },
  {
    id: 12,
    question: "Which prompt is likely to perform better?",
    options: {
      A: "Explain ML",
      B: "Explain ML in 3 bullet points for a beginner",
      C: "ML?",
      D: "Empty prompt"
    },
    correctAnswer: "B"
  },
  {
    id: 13,
    question: "Few-shot prompting involves:",
    options: {
      A: "No examples",
      B: "One example",
      C: "Multiple examples in prompt",
      D: "Fine-tuning model"
    },
    correctAnswer: "C"
  },
  {
    id: 14,
    question: "System prompts are used to:",
    options: {
      A: "Store memory",
      B: "Define model behavior & constraints",
      C: "Improve speed",
      D: "Reduce token cost"
    },
    correctAnswer: "B"
  },
  {
    id: 15,
    question: "Prompt engineering is important because:",
    options: {
      A: "Models are weak",
      B: "LLMs are sensitive to instruction framing",
      C: "It replaces fine-tuning",
      D: "It prevents all hallucinations"
    },
    correctAnswer: "B"
  },
  {
    id: 16,
    question: "Temperature controls:",
    options: {
      A: "Token limit",
      B: "Randomness of output",
      C: "Context size",
      D: "Model size"
    },
    correctAnswer: "B"
  },
  {
    id: 17,
    question: "Low temperature (e.g., 0.1) results in:",
    options: {
      A: "Creative answers",
      B: "Deterministic, focused output",
      C: "Hallucinations",
      D: "Longer responses"
    },
    correctAnswer: "B"
  },
  {
    id: 18,
    question: "High temperature (e.g., 0.9) results in:",
    options: {
      A: "Repetitive answers",
      B: "Less randomness",
      C: "More creative but less predictable output",
      D: "More accuracy"
    },
    correctAnswer: "C"
  },
  {
    id: 19,
    question: "Top-p (nucleus sampling) controls:",
    options: {
      A: "Maximum tokens",
      B: "Probability mass from which tokens are sampled",
      C: "Learning rate",
      D: "Prompt size"
    },
    correctAnswer: "B"
  },
  {
    id: 20,
    question: "In production systems, temperature is usually kept:",
    options: {
      A: "Very high",
      B: "Medium",
      C: "Low for consistency",
      D: "Random"
    },
    correctAnswer: "C"
  },
  {
    id: 21,
    question: "Hallucination means:",
    options: {
      A: "Model crashes",
      B: "Model generates false or made-up information",
      C: "Model refuses to answer",
      D: "Model repeats input"
    },
    correctAnswer: "B"
  },
  {
    id: 22,
    question: "Why do LLMs hallucinate?",
    options: {
      A: "They are badly trained",
      B: "They don’t have access to ground truth",
      C: "They optimize for fluent text, not factual correctness",
      D: "All of the above"
    },
    correctAnswer: "D"
  },
  {
    id: 23,
    question: "Hallucinations are more likely when:",
    options: {
      A: "Question is factual & grounded",
      B: "Context is missing or vague",
      C: "Temperature is low",
      D: "Using RAG"
    },
    correctAnswer: "B"
  },
  {
    id: 24,
    question: "Which does NOT help reduce hallucinations?",
    options: {
      A: "RAG",
      B: "Clear system prompt",
      C: "Lower temperature",
      D: "Asking model to “be confident”"
    },
    correctAnswer: "D"
  },
  {
    id: 25,
    question: "Why hallucinations are dangerous in production?",
    options: {
      A: "Poor grammar",
      B: "Misleading users with confident false answers",
      C: "Slow responses",
      D: "High token usage"
    },
    correctAnswer: "B"
  },
  {
    id: 26,
    question: "Prompting means:",
    options: {
      A: "Changing model weights",
      B: "Providing instructions at runtime",
      C: "Training model on custom data",
      D: "Storing embeddings"
    },
    correctAnswer: "B"
  },
  {
    id: 27,
    question: "Fine-tuning involves:",
    options: {
      A: "Prompt templates",
      B: "Updating model weights with domain data",
      C: "Vector databases",
      D: "Context windows"
    },
    correctAnswer: "B"
  },
  {
    id: 28,
    question: "RAG (Retrieval-Augmented Generation) combines:",
    options: {
      A: "Prompting + fine-tuning",
      B: "Search + LLM generation",
      C: "Training + inference",
      D: "Tokenization + decoding"
    },
    correctAnswer: "B"
  },
  {
    id: 29,
    question: "RAG is preferred when:",
    options: {
      A: "Data is static & small",
      B: "Data changes frequently and must be factual",
      C: "Creative writing",
      D: "Latency must be zero"
    },
    correctAnswer: "B"
  },
  {
    id: 30,
    question: "Fine-tuning is NOT ideal when:",
    options: {
      A: "You need domain style",
      B: "You need latest information updates frequently",
      C: "You control training data",
      D: "Data is stable"
    },
    correctAnswer: "B"
  },
  {
    id: 31,
    question: "Prompting is best when:",
    options: {
      A: "You need deep domain knowledge",
      B: "You need quick iteration & flexibility",
      C: "You want permanent memory",
      D: "Data is large"
    },
    correctAnswer: "B"
  },
  {
    id: 32,
    question: "RAG reduces hallucination by:",
    options: {
      A: "Reducing temperature",
      B: "Grounding responses in retrieved documents",
      C: "Increasing parameters",
      D: "Fine-tuning embeddings"
    },
    correctAnswer: "B"
  },
  {
    id: 33,
    question: "Which approach is most cost-effective initially?",
    options: {
      A: "Fine-tuning",
      B: "Training from scratch",
      C: "Prompting + RAG",
      D: "Custom LLM"
    },
    correctAnswer: "C"
  },
  {
    id: 34,
    question: "Embeddings are:",
    options: {
      A: "Tokens",
      B: "Numerical vector representations of text meaning",
      C: "Model weights",
      D: "Prompt templates"
    },
    correctAnswer: "B"
  },
  {
    id: 35,
    question: "Embeddings allow systems to:",
    options: {
      A: "Generate images",
      B: "Perform semantic search & similarity matching",
      C: "Fine-tune models",
      D: "Control temperature"
    },
    correctAnswer: "B"
  },
  {
    id: 36,
    question: "Two texts with similar meaning will have embeddings that are:",
    options: {
      A: "Identical",
      B: "Far apart",
      C: "Close in vector space",
      D: "Random"
    },
    correctAnswer: "C"
  },
  {
    id: 37,
    question: "Embeddings are commonly used in:",
    options: {
      A: "Classification only",
      B: "RAG systems, search, clustering",
      C: "Tokenization",
      D: "Prompt formatting"
    },
    correctAnswer: "B"
  },
  {
    id: 38,
    question: "Vector databases store:",
    options: {
      A: "Tokens",
      B: "Raw documents",
      C: "Embeddings for fast similarity search",
      D: "Model weights"
    },
    correctAnswer: "C"
  },
  {
    id: 39,
    question: "Why not just pass all documents in prompt instead of embeddings?",
    options: {
      A: "Slow model",
      B: "Token/context window limitations",
      C: "Higher accuracy",
      D: "Hallucinations"
    },
    correctAnswer: "B"
  },
  {
    id: 40,
    question: "You need chatbot answers strictly from company PDFs. Best approach:",
    options: {
      A: "Prompting only",
      B: "Fine-tuning",
      C: "RAG with embeddings",
      D: "Increase temperature"
    },
    correctAnswer: "C"
  },
  {
    id: 41,
    question: "You want model to follow company tone & style consistently:",
    options: {
      A: "RAG",
      B: "Fine-tuning",
      C: "Vector DB",
      D: "Top-p"
    },
    correctAnswer: "B"
  },
  {
    id: 42,
    question: "Model gives different answers every run. Fix:",
    options: {
      A: "Increase temperature",
      B: "Reduce temperature",
      C: "Increase top-p",
      D: "Add more tokens"
    },
    correctAnswer: "B"
  },
  {
    id: 43,
    question: "LLM confidently gives wrong legal advice. Root issue:",
    options: {
      A: "Model size",
      B: "Hallucination + lack of grounding",
      C: "Tokenization",
      D: "Prompt length"
    },
    correctAnswer: "B"
  },
  {
    id: 44,
    question: "Which improves factual accuracy most?",
    options: {
      A: "Larger model",
      B: "RAG + clear instructions",
      C: "High temperature",
      D: "Creative prompts"
    },
    correctAnswer: "B"
  },
  {
    id: 45,
    question: "Why embeddings are cheaper than fine-tuning?",
    options: {
      A: "No GPU needed",
      B: "No model weight updates required",
      C: "Faster training",
      D: "Smaller context window"
    },
    correctAnswer: "B"
  },
  {
    id: 46,
    question: "What happens if retrieved RAG documents are wrong?",
    options: {
      A: "Model corrects them",
      B: "Model likely amplifies incorrect info",
      C: "Model refuses",
      D: "Model ignores them"
    },
    correctAnswer: "B"
  },
  {
    id: 47,
    question: "LLMs do NOT inherently have:",
    options: {
      A: "Language understanding",
      B: "Reasoning ability",
      C: "Real-time knowledge access",
      D: "Pattern learning"
    },
    correctAnswer: "C"
  },
  {
    id: 48,
    question: "Which is most important for production LLM systems?",
    options: {
      A: "Creativity",
      B: "Controllability & grounding",
      C: "Model size",
      D: "Fancy prompts"
    },
    correctAnswer: "B"
  },
  {
    id: 49,
    question: "Why mid-level engineers must understand hallucinations?",
    options: {
      A: "For interviews",
      B: "For debugging real failures in production",
      C: "For research papers",
      D: "For training models"
    },
    correctAnswer: "B"
  },
  {
    id: 50,
    question: "A strong LLM engineer focuses on:",
    options: {
      A: "Prompt tricks only",
      B: "End-to-end system design (RAG, eval, safety)",
      C: "Training billion-parameter models",
      D: "UI design"
    },
    correctAnswer: "B"
  },
  {
  id: 51,
  question: "RAG is mainly used to:",
  options: {
    A: "Train LLMs faster",
    B: "Give LLMs access to external knowledge",
    C: "Increase token limits",
    D: "Replace vector databases"
  },
  correctAnswer: "B"
},
{
  id: 52,
  question: "The biggest limitation of standalone LLMs is:",
  options: {
    A: "Speed",
    B: "Lack of GPU support",
    C: "No access to private or up-to-date data",
    D: "Poor language understanding"
  },
  correctAnswer: "C"
},
{
  id: 53,
  question: "RAG helps solve which problem best?",
  options: {
    A: "Creativity",
    B: "Hallucinations due to missing context",
    C: "Tokenization",
    D: "Model fine-tuning"
  },
  correctAnswer: "B"
},
{
  id: 54,
  question: "RAG is preferred over fine-tuning when:",
  options: {
    A: "Data is static",
    B: "Data changes frequently",
    C: "Model must change behavior",
    D: "Data size is tiny"
  },
  correctAnswer: "B"
},
{
  id: 55,
  question: "Without RAG, enterprise LLM apps often:",
  options: {
    A: "Are faster",
    B: "Hallucinate confidently",
    C: "Cost less",
    D: "Need no validation"
  },
  correctAnswer: "B"
},
{
  id: 56,
  question: "Correct high-level RAG flow is:",
  options: {
    A: "Prompt → LLM → Database",
    B: "Query → LLM → Embeddings",
    C: "Query → Retrieve docs → LLM generates answer",
    D: "Docs → LLM → Query"
  },
  correctAnswer: "C"
},
{
  id: 57,
  question: "First step in RAG pipeline is:",
  options: {
    A: "Generate answer",
    B: "Create embeddings for documents",
    C: "Store tokens",
    D: "Fine-tune model"
  },
  correctAnswer: "B"
},
{
  id: 58,
  question: "At query time, user question is:",
  options: {
    A: "Sent directly to LLM",
    B: "Converted into embedding for similarity search",
    C: "Stored in vector DB",
    D: "Fine-tuned"
  },
  correctAnswer: "B"
},
{
  id: 59,
  question: "Retrieved documents are typically:",
  options: {
    A: "Replaced by model knowledge",
    B: "Injected into prompt context",
    C: "Stored permanently",
    D: "Used for training"
  },
  correctAnswer: "B"
},
{
  id: 60,
  question: "RAG is considered:",
  options: {
    A: "A model",
    B: "A system architecture pattern",
    C: "A training method",
    D: "An embedding algorithm"
  },
  correctAnswer: "B"
},
{
  id: 61,
  question: "Vector databases store:",
  options: {
    A: "Raw PDFs",
    B: "Tokens",
    C: "Embeddings + metadata",
    D: "Model weights"
  },
  correctAnswer: "C"
},
{
  id: 62,
  question: "FAISS is:",
  options: {
    A: "Cloud vector DB",
    B: "In-memory similarity search library by Meta",
    C: "LLM framework",
    D: "Data labeling tool"
  },
  correctAnswer: "B"
},
{
  id: 63,
  question: "Pinecone is:",
  options: {
    A: "Open-source library",
    B: "Managed cloud vector database",
    C: "LLM model",
    D: "Tokenizer"
  },
  correctAnswer: "B"
},
{
  id: 64,
  question: "Chroma is commonly used as:",
  options: {
    A: "Production-only DB",
    B: "Lightweight open-source vector DB for local/dev use",
    C: "Image DB",
    D: "Graph DB"
  },
  correctAnswer: "B"
},
{
  id: 65,
  question: "Vector DB is required because:",
  options: {
    A: "SQL is slow",
    B: "Text similarity needs nearest-neighbor search in high dimensions",
    C: "LLMs require it",
    D: "Tokens cannot be stored"
  },
  correctAnswer: "B"
},
{
  id: 66,
  question: "Chunking means:",
  options: {
    A: "Splitting models",
    B: "Splitting documents into smaller pieces",
    C: "Compressing embeddings",
    D: "Reducing tokens"
  },
  correctAnswer: "B"
},
{
  id: 67,
  question: "Why chunk documents before embedding?",
  options: {
    A: "Save storage",
    B: "Improve retrieval precision & context relevance",
    C: "Speed up training",
    D: "Reduce hallucinations directly"
  },
  correctAnswer: "B"
},
{
  id: 68,
  question: "Very large chunks lead to:",
  options: {
    A: "Better recall",
    B: "Irrelevant context being retrieved",
    C: "Faster search",
    D: "Smaller embeddings"
  },
  correctAnswer: "B"
},
{
  id: 69,
  question: "Very small chunks may cause:",
  options: {
    A: "Better understanding",
    B: "Loss of context & fragmented answers",
    C: "Lower storage",
    D: "Faster LLM"
  },
  correctAnswer: "B"
},
{
  id: 70,
  question: "Overlapping chunks help by:",
  options: {
    A: "Increasing cost only",
    B: "Preserving context across chunk boundaries",
    C: "Reducing tokens",
    D: "Eliminating hallucinations"
  },
  correctAnswer: "B"
},
{
  id: 71,
  question: "Ideal chunk size depends on:",
  options: {
    A: "Model name",
    B: "Document structure & query type",
    C: "GPU memory",
    D: "Vector DB"
  },
  correctAnswer: "B"
},
{
  id: 72,
  question: "Cosine similarity measures:",
  options: {
    A: "Distance in space",
    B: "Angle between vectors (direction similarity)",
    C: "Vector magnitude",
    D: "Token overlap"
  },
  correctAnswer: "B"
},
{
  id: 73,
  question: "Euclidean distance measures:",
  options: {
    A: "Angle",
    B: "Straight-line distance between vectors",
    C: "Probability",
    D: "Context window"
  },
  correctAnswer: "B"
},
{
  id: 74,
  question: "Cosine similarity is preferred when:",
  options: {
    A: "Vector magnitude matters",
    B: "Direction/semantic meaning matters",
    C: "Data is low-dimensional",
    D: "Using images only"
  },
  correctAnswer: "B"
},
{
  id: 75,
  question: "Embedding models usually normalize vectors because:",
  options: {
    A: "Faster DB",
    B: "Makes cosine similarity more meaningful",
    C: "Reduces tokens",
    D: "Saves memory"
  },
  correctAnswer: "B"
},
{
  id: 76,
  question: "Approximate Nearest Neighbor (ANN) search is used because:",
  options: {
    A: "Exact search is impossible",
    B: "Exact search is too slow at scale",
    C: "It improves accuracy",
    D: "LLMs require it"
  },
  correctAnswer: "B"
},
{
  id: 77,
  question: "RAG reduces hallucinations by:",
  options: {
    A: "Lowering temperature",
    B: "Grounding responses in retrieved documents",
    C: "Increasing context window",
    D: "Fine-tuning model"
  },
  correctAnswer: "B"
},
{
  id: 78,
  question: "LLM answers in RAG should be based on:",
  options: {
    A: "Model’s internal knowledge",
    B: "Retrieved context + question",
    C: "Prompt creativity",
    D: "Training data only"
  },
  correctAnswer: "B"
},
{
  id: 79,
  question: "If retrieved documents are empty:",
  options: {
    A: "Model still answers confidently",
    B: "Hallucination risk increases",
    C: "Accuracy improves",
    D: "Model stops"
  },
  correctAnswer: "B"
},
{
  id: 80,
  question: "Best practice to reduce hallucinations further:",
  options: {
    A: "Larger chunks",
    B: "Clear instruction: “Answer only from provided context”",
    C: "High temperature",
    D: "Remove system prompt"
  },
  correctAnswer: "B"
},
{
  id: 81,
  question: "Most common RAG failure is:",
  options: {
    A: "Model crash",
    B: "Poor retrieval quality",
    C: "Token overflow",
    D: "Fine-tuning issues"
  },
  correctAnswer: "B"
},
{
  id: 82,
  question: "If RAG retrieves irrelevant documents, model will:",
  options: {
    A: "Ignore them",
    B: "Hallucinate or answer incorrectly",
    C: "Ask clarification",
    D: "Refuse"
  },
  correctAnswer: "B"
},
{
  id: 83,
  question: "Embeddings trained on different domain cause:",
  options: {
    A: "Faster retrieval",
    B: "Semantic mismatch & poor search results",
    C: "Smaller vectors",
    D: "Better accuracy"
  },
  correctAnswer: "B"
},
{
  id: 84,
  question: "Chunking without metadata can cause:",
  options: {
    A: "Better answers",
    B: "Loss of source attribution & context",
    C: "Faster search",
    D: "Lower cost"
  },
  correctAnswer: "B"
},
{
  id: 85,
  question: "A RAG system answering outdated info usually means:",
  options: {
    A: "LLM issue",
    B: "Vector DB not updated/re-indexed",
    C: "Chunk size issue",
    D: "Temperature issue"
  },
  correctAnswer: "B"
},
{
  id: 86,
  question: "Increasing top-k retrieval blindly may:",
  options: {
    A: "Improve relevance always",
    B: "Add noise to prompt context",
    C: "Reduce hallucinations",
    D: "Reduce latency"
  },
  correctAnswer: "B"
},
{
  id: 87,
  question: "If context exceeds model limit:",
  options: {
    A: "Model uses all info",
    B: "Context gets truncated, losing important info",
    C: "Accuracy improves",
    D: "LLM compresses text"
  },
  correctAnswer: "B"
},
{
  id: 88,
  question: "Best RAG use case:",
  options: {
    A: "Poetry writing",
    B: "Internal company knowledge chatbot",
    C: "Image classification",
    D: "Sentiment analysis"
  },
  correctAnswer: "B"
},
{
  id: 89,
  question: "You need real-time policy updates in chatbot. Choose:",
  options: {
    A: "Fine-tuning",
    B: "RAG with frequently updated index",
    C: "Prompt engineering",
    D: "Larger model"
  },
  correctAnswer: "B"
},
{
  id: 90,
  question: "Model answers incorrectly even though data exists. First thing to check:",
  options: {
    A: "LLM temperature",
    B: "Retrieval quality & embeddings",
    C: "Model size",
    D: "UI"
  },
  correctAnswer: "B"
},
{
  id: 91,
  question: "Your RAG answers are verbose but wrong. Likely cause:",
  options: {
    A: "High temperature",
    B: "Retrieved context is irrelevant",
    C: "Chunk size small",
    D: "Vector DB slow"
  },
  correctAnswer: "B"
},
{
  id: 92,
  question: "Best way to debug RAG:",
  options: {
    A: "Change model",
    B: "Inspect retrieved chunks before generation",
    C: "Increase top-p",
    D: "Fine-tune"
  },
  correctAnswer: "B"
},
{
  id: 93,
  question: "Why citations are useful in RAG?",
  options: {
    A: "Improve UI",
    B: "Increase trust & debuggability",
    C: "Reduce cost",
    D: "Reduce tokens"
  },
  correctAnswer: "B"
},
{
  id: 94,
  question: "Multi-query retrieval improves:",
  options: {
    A: "Cost",
    B: "Recall for complex questions",
    C: "Token usage",
    D: "Latency"
  },
  correctAnswer: "B"
},
{
  id: 95,
  question: "Hybrid search combines:",
  options: {
    A: "Two LLMs",
    B: "Keyword (BM25) + vector search",
    C: "Prompt + fine-tuning",
    D: "FAISS + Pinecone"
  },
  correctAnswer: "B"
},
{
  id: 96,
  question: "RAG systems should log:",
  options: {
    A: "Only answers",
    B: "Queries, retrieved docs, and final prompt",
    C: "Model weights",
    D: "Token IDs"
  },
  correctAnswer: "B"
},
{
  id: 97,
  question: "Which does NOT directly improve RAG quality?",
  options: {
    A: "Better embeddings",
    B: "Better chunking",
    C: "Higher temperature",
    D: "Retrieval tuning"
  },
  correctAnswer: "C"
},
{
  id: 98,
  question: "RAG is NOT suitable when:",
  options: {
    A: "Data is private",
    B: "Data changes often",
    C: "Task is pure creativity (story writing)",
    D: "Accuracy is critical"
  },
  correctAnswer: "C"
},
{
  id: 99,
  question: "A strong RAG system is:",
  options: {
    A: "LLM-centric",
    B: "Retrieval-centric with controlled generation",
    C: "Prompt-centric only",
    D: "Fine-tuning heavy"
  },
  correctAnswer: "B"
},
{
  id: 100,
  question: "Mid-level engineers must understand RAG because:",
  options: {
    A: "It’s trendy",
    B: "Most real LLM products depend on it",
    C: "It replaces ML",
    D: "It removes need for databases"
  },
  correctAnswer: "B"
},
{
  id: 101,
  question: "Which statement best describes Artificial Intelligence (AI)?",
  options: {
    A: "A subset of Machine Learning",
    B: "A system that learns only from labeled data",
    C: "A broad field focused on making machines act intelligently",
    D: "Only neural-network-based systems"
  },
  correctAnswer: "C"
},
{
  id: 102,
  question: "Machine Learning differs from traditional programming because:",
  options: {
    A: "Rules are hard-coded",
    B: "Data is optional",
    C: "Models learn patterns from data instead of explicit rules",
    D: "It does not require algorithms"
  },
  correctAnswer: "C"
},
{
  id: 103,
  question: "Deep Learning is best described as:",
  options: {
    A: "Any model with more than one feature",
    B: "A subset of ML using multi-layer neural networks",
    C: "Rule-based AI",
    D: "Unsupervised learning only"
  },
  correctAnswer: "B"
},
{
  id: 104,
  question: "Which problem is best solved using Deep Learning rather than traditional ML?",
  options: {
    A: "Predicting house prices with small datasets",
    B: "Classifying handwritten digits from images",
    C: "Sorting numbers",
    D: "Linear regression with 10 rows"
  },
  correctAnswer: "B"
},
{
  id: 105,
  question: "Which of the following is NOT a requirement for Machine Learning?",
  options: {
    A: "Data",
    B: "Objective function",
    C: "Explicit rules for all scenarios",
    D: "Evaluation metric"
  },
  correctAnswer: "C"
},
{
  id: 106,
  question: "Predicting whether an email is spam or not is an example of:",
  options: {
    A: "Unsupervised learning",
    B: "Reinforcement learning",
    C: "Supervised learning",
    D: "Semi-supervised learning"
  },
  correctAnswer: "C"
},
{
  id: 107,
  question: "Customer segmentation without predefined labels is:",
  options: {
    A: "Supervised learning",
    B: "Unsupervised learning",
    C: "Semi-supervised learning",
    D: "Regression"
  },
  correctAnswer: "B"
},
{
  id: 108,
  question: "Which scenario best fits semi-supervised learning?",
  options: {
    A: "Fully labeled dataset",
    B: "No labeled data",
    C: "Small labeled + large unlabeled dataset",
    D: "Reinforcement environment"
  },
  correctAnswer: "C"
},
{
  id: 109,
  question: "Which algorithm is typically unsupervised?",
  options: {
    A: "Logistic Regression",
    B: "Random Forest",
    C: "K-Means",
    D: "Naive Bayes"
  },
  correctAnswer: "C"
},
{
  id: 110,
  question: "What is the main challenge in supervised learning?",
  options: {
    A: "Feature scaling",
    B: "Label availability and quality",
    C: "Choosing cluster count",
    D: "Lack of evaluation metrics"
  },
  correctAnswer: "B"
},
{
  id: 111,
  question: "High bias usually leads to:",
  options: {
    A: "Overfitting",
    B: "Underfitting",
    C: "High variance",
    D: "Data leakage"
  },
  correctAnswer: "B"
},
{
  id: 112,
  question: "High variance indicates that the model:",
  options: {
    A: "Is too simple",
    B: "Performs well on unseen data",
    C: "Is too complex and overfits",
    D: "Has high bias"
  },
  correctAnswer: "C"
},
{
  id: 113,
  question: "Which model is more likely to have high bias?",
  options: {
    A: "Deep neural network",
    B: "Random Forest",
    C: "Linear regression on complex data",
    D: "XGBoost"
  },
  correctAnswer: "C"
},
{
  id: 114,
  question: "Increasing training data usually helps reduce:",
  options: {
    A: "Bias only",
    B: "Variance only",
    C: "Both equally",
    D: "Neither"
  },
  correctAnswer: "B"
},
{
  id: 115,
  question: "Bias–Variance tradeoff exists because:",
  options: {
    A: "Data is always noisy",
    B: "More complex models always win",
    C: "Reducing one often increases the other",
    D: "Metrics are inaccurate"
  },
  correctAnswer: "C"
},
{
  id: 116,
  question: "A model performs very well on training data but poorly on test data. This is:",
  options: {
    A: "Underfitting",
    B: "Overfitting",
    C: "Data leakage",
    D: "Proper generalization"
  },
  correctAnswer: "B"
},
{
  id: 117,
  question: "Which is a sign of underfitting?",
  options: {
    A: "Low training error, high test error",
    B: "High training error, high test error",
    C: "Low training and test error",
    D: "Zero loss"
  },
  correctAnswer: "B"
},
{
  id: 118,
  question: "Which technique helps reduce overfitting?",
  options: {
    A: "Adding more features blindly",
    B: "Increasing model complexity",
    C: "Regularization",
    D: "Removing validation set"
  },
  correctAnswer: "C"
},
{
  id: 119,
  question: "Deep learning models overfit more easily because:",
  options: {
    A: "They use GPUs",
    B: "They have many parameters",
    C: "They are unsupervised",
    D: "They don’t use loss functions"
  },
  correctAnswer: "B"
},
{
  id: 120,
  question: "If both training and validation errors are high, what is the issue?",
  options: {
    A: "Overfitting",
    B: "Data leakage",
    C: "Underfitting",
    D: "Class imbalance"
  },
  correctAnswer: "C"
},
{
  id: 121,
  question: "Why do we need a validation set?",
  options: {
    A: "To increase training accuracy",
    B: "To tune hyperparameters",
    C: "To deploy the model",
    D: "To store logs"
  },
  correctAnswer: "B"
},
{
  id: 122,
  question: "Which data should never be used during model training or tuning?",
  options: {
    A: "Training set",
    B: "Validation set",
    C: "Test set",
    D: "Augmented data"
  },
  correctAnswer: "C"
},
{
  id: 123,
  question: "Typical train/validation/test split is:",
  options: {
    A: "100/0/0",
    B: "80/10/10 or 70/15/15",
    C: "50/50/0",
    D: "10/80/10"
  },
  correctAnswer: "B"
},
{
  id: 124,
  question: "Using test data to tune hyperparameters leads to:",
  options: {
    A: "Better generalization",
    B: "Data leakage",
    C: "Regularization",
    D: "Lower bias"
  },
  correctAnswer: "B"
},
{
  id: 125,
  question: "Cross-validation mainly helps when:",
  options: {
    A: "Dataset is large",
    B: "Dataset is small",
    C: "Model is linear",
    D: "Data is unlabeled"
  },
  correctAnswer: "B"
},
{
  id: 126,
  question: "Feature engineering mainly aims to:",
  options: {
    A: "Increase dataset size",
    B: "Improve model performance using better inputs",
    C: "Reduce training time only",
    D: "Remove labels"
  },
  correctAnswer: "B"
},
{
  id: 127,
  question: "Which is an example of feature engineering?",
  options: {
    A: "Changing algorithm",
    B: "Normalizing numerical values",
    C: "Increasing epochs",
    D: "Hyperparameter tuning"
  },
  correctAnswer: "B"
},
{
  id: 128,
  question: "Why is feature scaling important?",
  options: {
    A: "Improves data quality",
    B: "Helps gradient-based algorithms converge faster",
    C: "Removes outliers",
    D: "Prevents data leakage"
  },
  correctAnswer: "B"
},
{
  id: 129,
  question: "One-hot encoding is used for:",
  options: {
    A: "Numerical features",
    B: "Ordinal data only",
    C: "Categorical features",
    D: "Target variables"
  },
  correctAnswer: "C"
},
{
  id: 130,
  question: "Poor feature engineering usually results in:",
  options: {
    A: "Faster training",
    B: "Better generalization",
    C: "Lower model performance",
    D: "Higher accuracy"
  },
  correctAnswer: "C"
},
{
  id: 131,
  question: "Data leakage occurs when:",
  options: {
    A: "Dataset is too large",
    B: "Training data is noisy",
    C: "Future information is used during training",
    D: "Model is too complex"
  },
  correctAnswer: "C"
},
{
  id: 132,
  question: "Which is a common cause of data leakage?",
  options: {
    A: "Shuffling data",
    B: "Scaling entire dataset before splitting",
    C: "Using cross-validation",
    D: "Feature selection"
  },
  correctAnswer: "B"
},
{
  id: 133,
  question: "Why is data leakage dangerous?",
  options: {
    A: "It slows training",
    B: "It gives overly optimistic results",
    C: "It increases bias",
    D: "It reduces dataset size"
  },
  correctAnswer: "B"
},
{
  id: 134,
  question: "Leakage usually causes:",
  options: {
    A: "Underfitting",
    B: "Poor training accuracy",
    C: "Unrealistically high validation scores",
    D: "Class imbalance"
  },
  correctAnswer: "C"
},
{
  id: 135,
  question: "Which practice prevents data leakage?",
  options: {
    A: "Scaling after splitting data",
    B: "Using test data early",
    C: "Feature selection on full data",
    D: "Mixing datasets"
  },
  correctAnswer: "A"
},
{
  id: 136,
  question: "A model trained on future sales data predicts past sales accurately. Issue?",
  options: {
    A: "Overfitting",
    B: "Bias",
    C: "Data leakage",
    D: "Underfitting"
  },
  correctAnswer: "C"
},
{
  id: 137,
  question: "Which learning type is used in recommendation systems without explicit labels?",
  options: {
    A: "Supervised",
    B: "Unsupervised",
    C: "Semi-supervised",
    D: "Reinforcement"
  },
  correctAnswer: "B"
},
{
  id: 138,
  question: "High bias models are usually:",
  options: {
    A: "Very complex",
    B: "Hard to interpret",
    C: "Too simple",
    D: "Data-hungry"
  },
  correctAnswer: "C"
},
{
  id: 139,
  question: "Increasing model complexity generally:",
  options: {
    A: "Increases bias",
    B: "Decreases variance",
    C: "Decreases bias but increases variance",
    D: "Reduces both"
  },
  correctAnswer: "C"
},
{
  id: 140,
  question: "Feature created using target variable during training causes:",
  options: {
    A: "Regularization",
    B: "Overfitting",
    C: "Data leakage",
    D: "Underfitting"
  },
  correctAnswer: "C"
},
{
  id: 141,
  question: "Which dataset best represents real-world performance?",
  options: {
    A: "Training",
    B: "Validation",
    C: "Test",
    D: "Augmented"
  },
  correctAnswer: "C"
},
{
  id: 142,
  question: "Why should feature engineering pipelines be consistent in training & inference?",
  options: {
    A: "For speed",
    B: "To avoid data leakage and mismatch",
    C: "For logging",
    D: "For hyperparameter tuning"
  },
  correctAnswer: "B"
},
{
  id: 143,
  question: "Using customer future churn status as a feature to predict churn is:",
  options: {
    A: "Feature selection",
    B: "Bias",
    C: "Data leakage",
    D: "Overfitting"
  },
  correctAnswer: "C"
},
{
  id: 144,
  question: "A model with low bias and low variance is:",
  options: {
    A: "Impossible",
    B: "Ideal but rare",
    C: "Always deep learning",
    D: "Always linear"
  },
  correctAnswer: "B"
},
{
  id: 145,
  question: "Which technique helps detect overfitting early?",
  options: {
    A: "Increasing epochs",
    B: "Monitoring validation loss",
    C: "Removing test set",
    D: "Feature scaling"
  },
  correctAnswer: "B"
},
{
  id: 146,
  question: "If validation performance fluctuates heavily, likely cause is:",
  options: {
    A: "High bias",
    B: "High variance",
    C: "Data leakage",
    D: "Proper training"
  },
  correctAnswer: "B"
},
{
  id: 147,
  question: "Which learning type does NOT require labeled data?",
  options: {
    A: "Supervised",
    B: "Unsupervised",
    C: "Semi-supervised",
    D: "Regression"
  },
  correctAnswer: "B"
},
{
  id: 148,
  question: "Overfitting can be reduced by:",
  options: {
    A: "More parameters",
    B: "Less data",
    C: "Regularization & dropout",
    D: "Removing validation set"
  },
  correctAnswer: "C"
},
{
  id: 149,
  question: "Which step should ALWAYS be done after data split?",
  options: {
    A: "Feature scaling",
    B: "Label encoding",
    C: "Data preprocessing pipelines per split",
    D: "Feature selection on full data"
  },
  correctAnswer: "C"
},
{
  id: 150,
  question: "The biggest red flag in an ML experiment is:",
  options: {
    A: "Low accuracy",
    B: "Slow training",
    C: "Near-perfect validation score early on",
    D: "Small dataset"
  },
  correctAnswer: "C"
},
{
  id: 151,
  question: "Linear Regression is best suited when:",
  options: {
    A: "Target is categorical",
    B: "Relationship between features and target is approximately linear",
    C: "Data is highly non-linear",
    D: "Labels are missing"
  },
  correctAnswer: "B"
},
{
  id: 152,
  question: "Logistic Regression is preferred over Linear Regression when:",
  options: {
    A: "Predicting continuous values",
    B: "Predicting probabilities of classes",
    C: "Dataset is very large",
    D: "Features are unscaled"
  },
  correctAnswer: "B"
},
{
  id: 153,
  question: "Logistic Regression outputs:",
  options: {
    A: "Class labels directly",
    B: "Continuous unbounded values",
    C: "Probabilities between 0 and 1",
    D: "Cluster assignments"
  },
  correctAnswer: "C"
},
{
  id: 154,
  question: "One major advantage of Logistic Regression is:",
  options: {
    A: "Handles non-linearity automatically",
    B: "High interpretability of features",
    C: "Works without labeled data",
    D: "No need for preprocessing"
  },
  correctAnswer: "B"
},
{
  id: 155,
  question: "Linear Regression performs poorly when:",
  options: {
    A: "Features are correlated",
    B: "Data is small",
    C: "Relationship is highly non-linear",
    D: "Output is numerical"
  },
  correctAnswer: "C"
},
{
  id: 156,
  question: "Decision Trees are popular because they:",
  options: {
    A: "Require feature scaling",
    B: "Are easy to interpret",
    C: "Never overfit",
    D: "Only work on numerical data"
  },
  correctAnswer: "B"
},
{
  id: 157,
  question: "A major drawback of Decision Trees is:",
  options: {
    A: "Low bias",
    B: "High variance (overfitting)",
    C: "Inability to handle missing values",
    D: "Slow prediction time"
  },
  correctAnswer: "B"
},
{
  id: 158,
  question: "Decision Trees handle categorical features:",
  options: {
    A: "Poorly",
    B: "Only with encoding",
    C: "Naturally (in many implementations)",
    D: "Not at all"
  },
  correctAnswer: "C"
},
{
  id: 159,
  question: "Increasing tree depth usually:",
  options: {
    A: "Reduces variance",
    B: "Increases bias",
    C: "Increases overfitting risk",
    D: "Improves generalization always"
  },
  correctAnswer: "C"
},
{
  id: 160,
  question: "Decision Trees are especially useful when:",
  options: {
    A: "Data is extremely large and linear",
    B: "Interpretability is important",
    C: "Dataset has only numeric data",
    D: "Data is unlabeled"
  },
  correctAnswer: "B"
},
{
  id: 161,
  question: "Random Forest improves Decision Trees by:",
  options: {
    A: "Using one deep tree",
    B: "Averaging multiple trees to reduce variance",
    C: "Reducing bias only",
    D: "Removing randomness"
  },
  correctAnswer: "B"
},
{
  id: 162,
  question: "Random Forest works best when:",
  options: {
    A: "Data is very small",
    B: "Features are highly correlated",
    C: "Overfitting is a concern with trees",
    D: "Interpretability is not needed"
  },
  correctAnswer: "C"
},
{
  id: 163,
  question: "Which statement about Random Forest is TRUE?",
  options: {
    A: "Always faster than single tree",
    B: "Less overfitting than a single tree",
    C: "Cannot handle missing values",
    D: "Requires feature scaling"
  },
  correctAnswer: "B"
},
{
  id: 164,
  question: "XGBoost differs from Random Forest because it:",
  options: {
    A: "Builds trees independently",
    B: "Uses boosting (sequential learning)",
    C: "Is unsupervised",
    D: "Cannot handle missing data"
  },
  correctAnswer: "B"
},
{
  id: 165,
  question: "XGBoost is often preferred when:",
  options: {
    A: "Dataset is very small",
    B: "High performance is required in competitions/production",
    C: "Interpretability is top priority",
    D: "Labels are missing"
  },
  correctAnswer: "B"
},
{
  id: 166,
  question: "A downside of XGBoost is:",
  options: {
    A: "Poor accuracy",
    B: "High bias",
    C: "More complex tuning required",
    D: "Cannot handle non-linearity"
  },
  correctAnswer: "C"
},
{
  id: 167,
  question: "KNN is a:",
  options: {
    A: "Parametric model",
    B: "Lazy learner (no training phase)",
    C: "Tree-based model",
    D: "Linear classifier"
  },
  correctAnswer: "B"
},
{
  id: 168,
  question: "KNN performs poorly when:",
  options: {
    A: "Dataset is small",
    B: "Dimensionality is high",
    C: "Data is normalized",
    D: "Classes are balanced"
  },
  correctAnswer: "B"
},
{
  id: 169,
  question: "Choosing very small K value may cause:",
  options: {
    A: "Underfitting",
    B: "High bias",
    C: "Overfitting",
    D: "Low variance"
  },
  correctAnswer: "C"
},
{
  id: 170,
  question: "Choosing very large K value may cause:",
  options: {
    A: "Overfitting",
    B: "High variance",
    C: "Underfitting",
    D: "Noise sensitivity"
  },
  correctAnswer: "C"
},
{
  id: 171,
  question: "KNN requires feature scaling because:",
  options: {
    A: "It uses gradient descent",
    B: "Distance calculation is sensitive to scale",
    C: "It uses probability outputs",
    D: "It is tree-based"
  },
  correctAnswer: "B"
},
{
  id: 172,
  question: "SVM tries to:",
  options: {
    A: "Minimize error only",
    B: "Maximize margin between classes",
    C: "Reduce variance only",
    D: "Cluster data"
  },
  correctAnswer: "B"
},
{
  id: 173,
  question: "SVM works best when:",
  options: {
    A: "Dataset is huge",
    B: "Data is linearly separable or nearly separable",
    C: "Features are not scaled",
    D: "Interpretability is required"
  },
  correctAnswer: "B"
},
{
  id: 174,
  question: "Kernel trick in SVM helps to:",
  options: {
    A: "Reduce training time",
    B: "Handle non-linear boundaries",
    C: "Avoid scaling",
    D: "Reduce dimensionality"
  },
  correctAnswer: "B"
},
{
  id: 175,
  question: "Which kernel is commonly used for non-linear data?",
  options: {
    A: "Linear",
    B: "Polynomial",
    C: "RBF (Gaussian)",
    D: "Sigmoid only"
  },
  correctAnswer: "C"
},
{
  id: 176,
  question: "A drawback of SVM is:",
  options: {
    A: "Poor accuracy",
    B: "High memory usage on large datasets",
    C: "Inability to handle non-linear data",
    D: "No hyperparameters"
  },
  correctAnswer: "B"
},
{
  id: 177,
  question: "Naive Bayes is called “naive” because it:",
  options: {
    A: "Is simple",
    B: "Assumes feature independence",
    C: "Uses probability",
    D: "Works only on text"
  },
  correctAnswer: "B"
},
{
  id: 178,
  question: "Naive Bayes works particularly well for:",
  options: {
    A: "Image classification",
    B: "Regression problems",
    C: "Text classification (spam, sentiment)",
    D: "Time-series forecasting"
  },
  correctAnswer: "C"
},
{
  id: 179,
  question: "One advantage of Naive Bayes is:",
  options: {
    A: "High accuracy always",
    B: "Requires large datasets",
    C: "Fast and works well with high-dimensional data",
    D: "Handles feature correlation well"
  },
  correctAnswer: "C"
},
{
  id: 180,
  question: "Naive Bayes performs poorly when:",
  options: {
    A: "Features are independent",
    B: "Dataset is small",
    C: "Features are highly correlated",
    D: "Data is categorical"
  },
  correctAnswer: "C"
},
{
  id: 181,
  question: "K-Means clustering requires:",
  options: {
    A: "Labeled data",
    B: "Predefined number of clusters (K)",
    C: "Distance metric only",
    D: "Non-numeric data"
  },
  correctAnswer: "B"
},
{
  id: 182,
  question: "K-Means performs poorly when:",
  options: {
    A: "Clusters are spherical",
    B: "Data is numeric",
    C: "Clusters have varying densities or shapes",
    D: "Dataset is small"
  },
  correctAnswer: "C"
},
{
  id: 183,
  question: "DBSCAN differs from K-Means because it:",
  options: {
    A: "Needs K value",
    B: "Detects noise/outliers naturally",
    C: "Works only on small datasets",
    D: "Requires labels"
  },
  correctAnswer: "B"
},
{
  id: 184,
  question: "DBSCAN is best suited when:",
  options: {
    A: "Clusters are well-separated and spherical",
    B: "Noise and outliers are important to detect",
    C: "Data is very high dimensional",
    D: "K is known"
  },
  correctAnswer: "B"
},
{
  id: 185,
  question: "A limitation of DBSCAN is:",
  options: {
    A: "Cannot detect noise",
    B: "Struggles with varying density clusters",
    C: "Requires labeled data",
    D: "Uses centroids"
  },
  correctAnswer: "B"
},
{
  id: 186,
  question: "PCA is mainly used to:",
  options: {
    A: "Increase features",
    B: "Reduce dimensionality while preserving variance",
    C: "Improve labels",
    D: "Cluster data"
  },
  correctAnswer: "B"
},
{
  id: 187,
  question: "PCA works by:",
  options: {
    A: "Selecting random features",
    B: "Creating new orthogonal features (components)",
    C: "Removing correlated features only",
    D: "Label encoding"
  },
  correctAnswer: "B"
},
{
  id: 188,
  question: "PCA is helpful when:",
  options: {
    A: "Dataset has few features",
    B: "Data has multicollinearity",
    C: "Labels are missing",
    D: "Model is linear"
  },
  correctAnswer: "B"
},
{
  id: 189,
  question: "PCA may hurt performance when:",
  options: {
    A: "Data is noisy",
    B: "Interpretability is critical",
    C: "Features are correlated",
    D: "Dimensionality is high"
  },
  correctAnswer: "B"
},
{
  id: 190,
  question: "PCA should be applied:",
  options: {
    A: "Before train-test split",
    B: "After train-test split using training data only",
    C: "On full dataset",
    D: "On test data only"
  },
  correctAnswer: "B"
},
{
  id: 191,
  question: "You need a fast, interpretable binary classifier. Best choice:",
  options: {
    A: "XGBoost",
    B: "Logistic Regression",
    C: "KNN",
    D: "DBSCAN"
  },
  correctAnswer: "B"
},
{
  id: 192,
  question: "You have non-linear data and small dataset. Best option:",
  options: {
    A: "Linear Regression",
    B: "SVM with RBF kernel",
    C: "K-Means",
    D: "Naive Bayes"
  },
  correctAnswer: "B"
},
{
  id: 193,
  question: "Large dataset, mixed features, strong performance needed:",
  options: {
    A: "KNN",
    B: "Decision Tree",
    C: "Random Forest / XGBoost",
    D: "Linear Regression"
  },
  correctAnswer: "C"
},
{
  id: 194,
  question: "Text classification with limited data:",
  options: {
    A: "SVM only",
    B: "Naive Bayes",
    C: "K-Means",
    D: "PCA"
  },
  correctAnswer: "B"
},
{
  id: 195,
  question: "Detecting fraudulent transactions as outliers:",
  options: {
    A: "K-Means",
    B: "DBSCAN",
    C: "Logistic Regression",
    D: "PCA"
  },
  correctAnswer: "B"
},
{
  id: 196,
  question: "Model overfits heavily; which algorithm helps?",
  options: {
    A: "Single Decision Tree",
    B: "Random Forest",
    C: "KNN with K=1",
    D: "Deep neural network"
  },
  correctAnswer: "B"
},
{
  id: 197,
  question: "Which model requires least training time?",
  options: {
    A: "XGBoost",
    B: "Random Forest",
    C: "KNN (training phase)",
    D: "SVM"
  },
  correctAnswer: "C"
},
{
  id: 198,
  question: "Which algorithm is most sensitive to feature scaling?",
  options: {
    A: "Decision Tree",
    B: "Random Forest",
    C: "KNN",
    D: "Naive Bayes"
  },
  correctAnswer: "C"
},
{
  id: 199,
  question: "PCA is NOT suitable when:",
  options: {
    A: "Dimensionality is high",
    B: "Features are correlated",
    C: "Model explainability is required",
    D: "Training is slow"
  },
  correctAnswer: "C"
},
{
  id: 200,
  question: "Best algorithm for a quick baseline model:",
  options: {
    A: "XGBoost",
    B: "Logistic Regression / Naive Bayes",
    C: "DBSCAN",
    D: "Deep Learning"
  },
  correctAnswer: "B"
},
{
  id: 201,
  question: "Accuracy is a good metric when:",
  options: {
    A: "Classes are highly imbalanced",
    B: "False positives are very costly",
    C: "Dataset is balanced and errors have equal cost",
    D: "Only minority class matters"
  },
  correctAnswer: "C"
},
{
  id: 202,
  question: "Precision measures:",
  options: {
    A: "How many actual positives were correctly predicted",
    B: "How many predicted positives are actually correct",
    C: "Overall correctness",
    D: "True negatives rate"
  },
  correctAnswer: "B"
},
{
  id: 203,
  question: "Recall focuses on:",
  options: {
    A: "Minimizing false positives",
    B: "Maximizing true negatives",
    C: "Capturing as many actual positives as possible",
    D: "Overall accuracy"
  },
  correctAnswer: "C"
},
{
  id: 204,
  question: "High precision but low recall means:",
  options: {
    A: "Many false positives",
    B: "Many false negatives",
    C: "Balanced model",
    D: "Overfitting"
  },
  correctAnswer: "B"
},
{
  id: 205,
  question: "High recall but low precision means:",
  options: {
    A: "Many false positives",
    B: "Many false negatives",
    C: "Low variance",
    D: "Underfitting"
  },
  correctAnswer: "A"
},
{
  id: 206,
  question: "F1-score is best described as:",
  options: {
    A: "Average of precision and recall",
    B: "Harmonic mean of precision and recall",
    C: "Difference of precision and recall",
    D: "Weighted accuracy"
  },
  correctAnswer: "B"
},
{
  id: 207,
  question: "F1-score is most useful when:",
  options: {
    A: "Dataset is balanced",
    B: "Only accuracy matters",
    C: "Classes are imbalanced and both FP & FN matter",
    D: "Only recall matters"
  },
  correctAnswer: "C"
},
{
  id: 208,
  question: "Which metric ignores true negatives?",
  options: {
    A: "Accuracy",
    B: "Precision",
    C: "Recall",
    D: "F1-score"
  },
  correctAnswer: "D"
},
{
  id: 209,
  question: "In fraud detection, which metric is usually prioritized?",
  options: {
    A: "Accuracy",
    B: "Precision",
    C: "Recall",
    D: "R²"
  },
  correctAnswer: "C"
},
{
  id: 210,
  question: "In email spam detection, high precision ensures:",
  options: {
    A: "All spam is caught",
    B: "Fewer genuine emails marked as spam",
    C: "Higher recall",
    D: "More false negatives"
  },
  correctAnswer: "B"
},
{
  id: 211,
  question: "A false positive means:",
  options: {
    A: "Model predicted negative, actual positive",
    B: "Model predicted positive, actual negative",
    C: "Model predicted correctly",
    D: "Model is biased"
  },
  correctAnswer: "B"
},
{
  id: 212,
  question: "A false negative means:",
  options: {
    A: "Model missed a positive case",
    B: "Model predicted positive incorrectly",
    C: "Model is overfitting",
    D: "Model predicted correctly"
  },
  correctAnswer: "A"
},
{
  id: 213,
  question: "In medical diagnosis, false negatives are dangerous because:",
  options: {
    A: "Healthy patients get treatment",
    B: "Sick patients are missed",
    C: "Accuracy drops",
    D: "Precision increases"
  },
  correctAnswer: "B"
},
{
  id: 214,
  question: "Which value is on the diagonal of a confusion matrix?",
  options: {
    A: "Errors only",
    B: "Correct predictions only",
    C: "False predictions",
    D: "Probabilities"
  },
  correctAnswer: "B"
},
{
  id: 215,
  question: "Which confusion matrix cell is most critical for fraud detection?",
  options: {
    A: "True negatives",
    B: "False positives",
    C: "False negatives",
    D: "True positives"
  },
  correctAnswer: "C"
},
{
  id: 216,
  question: "ROC curve plots:",
  options: {
    A: "Precision vs Recall",
    B: "Accuracy vs Threshold",
    C: "True Positive Rate vs False Positive Rate",
    D: "Recall vs F1"
  },
  correctAnswer: "C"
},
{
  id: 217,
  question: "ROC-AUC represents:",
  options: {
    A: "Accuracy at one threshold",
    B: "Average loss",
    C: "Model’s ability to separate classes across thresholds",
    D: "Probability calibration"
  },
  correctAnswer: "C"
},
{
  id: 218,
  question: "A random classifier has ROC-AUC of:",
  options: {
    A: "0.0",
    B: "0.3",
    C: "0.5",
    D: "1.0"
  },
  correctAnswer: "C"
},
{
  id: 219,
  question: "A perfect classifier has ROC-AUC of:",
  options: {
    A: "0.5",
    B: "0.7",
    C: "0.9",
    D: "1.0"
  },
  correctAnswer: "D"
},
{
  id: 220,
  question: "ROC-AUC is especially useful when:",
  options: {
    A: "Dataset is very small",
    B: "Classes are imbalanced and threshold varies",
    C: "Only one threshold matters",
    D: "Output is regression"
  },
  correctAnswer: "B"
},
{
  id: 221,
  question: "MAE measures:",
  options: {
    A: "Average squared error",
    B: "Average absolute error magnitude",
    C: "Variance of error",
    D: "Correlation"
  },
  correctAnswer: "B"
},
{
  id: 222,
  question: "MSE differs from MAE because it:",
  options: {
    A: "Penalizes large errors more heavily",
    B: "Ignores outliers",
    C: "Is easier to interpret",
    D: "Is scale-free"
  },
  correctAnswer: "A"
},
{
  id: 223,
  question: "RMSE is preferred when:",
  options: {
    A: "You want squared units",
    B: "Large errors are unacceptable and interpretability matters",
    C: "Data has no outliers",
    D: "Scale doesn’t matter"
  },
  correctAnswer: "B"
},
{
  id: 224,
  question: "Which regression metric is most sensitive to outliers?",
  options: {
    A: "MAE",
    B: "MSE",
    C: "R²",
    D: "Median error"
  },
  correctAnswer: "B"
},
{
  id: 225,
  question: "R² represents:",
  options: {
    A: "Average error",
    B: "Model accuracy",
    C: "Proportion of variance explained by the model",
    D: "Correlation only"
  },
  correctAnswer: "C"
},
{
  id: 226,
  question: "R² can be negative when:",
  options: {
    A: "Model is perfect",
    B: "Model performs worse than predicting mean",
    C: "Data is small",
    D: "Errors are symmetric"
  },
  correctAnswer: "B"
},
{
  id: 227,
  question: "Which metric is easiest to explain to business stakeholders?",
  options: {
    A: "MSE",
    B: "RMSE",
    C: "MAE",
    D: "R²"
  },
  correctAnswer: "C"
},
{
  id: 228,
  question: "RMSE is measured in:",
  options: {
    A: "Squared units",
    B: "Same units as target variable",
    C: "Percentage",
    D: "Unitless"
  },
  correctAnswer: "B"
},
{
  id: 229,
  question: "Credit default prediction should prioritize:",
  options: {
    A: "Accuracy",
    B: "Precision",
    C: "Recall (catch defaulters)",
    D: "R²"
  },
  correctAnswer: "C"
},
{
  id: 230,
  question: "Email spam filtering should prioritize:",
  options: {
    A: "Recall",
    B: "Precision",
    C: "Accuracy",
    D: "ROC only"
  },
  correctAnswer: "B"
},
{
  id: 231,
  question: "Medical cancer screening typically prioritizes:",
  options: {
    A: "Precision",
    B: "Recall",
    C: "Accuracy",
    D: "F1 only"
  },
  correctAnswer: "B"
},
{
  id: 232,
  question: "Recommendation systems often evaluate using:",
  options: {
    A: "Accuracy",
    B: "MAE/RMSE for ratings",
    C: "ROC only",
    D: "Confusion matrix"
  },
  correctAnswer: "B"
},
{
  id: 233,
  question: "When false positives and false negatives have equal cost, use:",
  options: {
    A: "Precision",
    B: "Recall",
    C: "Accuracy or F1",
    D: "ROC-AUC only"
  },
  correctAnswer: "C"
},
{
  id: 234,
  question: "Why accuracy is misleading for imbalanced datasets?",
  options: {
    A: "It ignores true positives",
    B: "Majority class dominates score",
    C: "It is hard to compute",
    D: "It depends on threshold"
  },
  correctAnswer: "B"
},
{
  id: 235,
  question: "Cross-validation helps to:",
  options: {
    A: "Increase training data",
    B: "Reduce overfitting estimation bias",
    C: "Speed up training",
    D: "Remove need for test set"
  },
  correctAnswer: "B"
},
{
  id: 236,
  question: "K-Fold cross-validation means:",
  options: {
    A: "K models trained independently on full data",
    B: "Data split into K parts; each used as validation once",
    C: "K different algorithms",
    D: "K test sets"
  },
  correctAnswer: "B"
},
{
  id: 237,
  question: "Cross-validation is most useful when:",
  options: {
    A: "Dataset is very large",
    B: "Dataset is small",
    C: "Model is simple",
    D: "Data is unlabeled"
  },
  correctAnswer: "B"
},
{
  id: 238,
  question: "Which data should NEVER be used in cross-validation?",
  options: {
    A: "Training data",
    B: "Validation folds",
    C: "Test data",
    D: "Features"
  },
  correctAnswer: "C"
},
{
  id: 239,
  question: "Stratified cross-validation ensures:",
  options: {
    A: "Equal fold sizes",
    B: "Same class distribution in each fold",
    C: "Faster training",
    D: "Less variance"
  },
  correctAnswer: "B"
},
{
  id: 240,
  question: "Cross-validation mainly helps estimate:",
  options: {
    A: "Training loss",
    B: "Model generalization performance",
    C: "Feature importance",
    D: "Hyperparameters only"
  },
  correctAnswer: "B"
},
{
  id: 241,
  question: "Model has high ROC-AUC but poor precision. Why?",
  options: {
    A: "Model is random",
    B: "Threshold selection issue",
    C: "Data leakage",
    D: "Underfitting"
  },
  correctAnswer: "B"
},
{
  id: 242,
  question: "Business cares more about avoiding missed fraud than blocking genuine users. Best metric:",
  options: {
    A: "Precision",
    B: "Accuracy",
    C: "Recall",
    D: "R²"
  },
  correctAnswer: "C"
},
{
  id: 243,
  question: "Model evaluation score drops significantly from CV to test set. Likely reason:",
  options: {
    A: "Better generalization",
    B: "Data leakage during CV",
    C: "Overfitting to CV folds",
    D: "Balanced data"
  },
  correctAnswer: "C"
},
{
  id: 244,
  question: "Which metric compares models independent of classification threshold?",
  options: {
    A: "Accuracy",
    B: "Precision",
    C: "ROC-AUC",
    D: "F1"
  },
  correctAnswer: "C"
},
{
  id: 245,
  question: "MAE is preferred over RMSE when:",
  options: {
    A: "Large errors matter more",
    B: "Outliers exist and should not dominate",
    C: "Squared penalties needed",
    D: "Units don’t matter"
  },
  correctAnswer: "B"
},
{
  id: 246,
  question: "Which metric helps choose optimal threshold?",
  options: {
    A: "R²",
    B: "Confusion matrix",
    C: "RMSE",
    D: "MAE"
  },
  correctAnswer: "B"
},
{
  id: 247,
  question: "High recall but poor business outcome suggests:",
  options: {
    A: "Model is perfect",
    B: "Too many false positives hurting operations",
    C: "Data leakage",
    D: "High bias"
  },
  correctAnswer: "B"
},
{
  id: 248,
  question: "Which metric should be monitored post-deployment?",
  options: {
    A: "Training accuracy",
    B: "Validation loss",
    C: "Business-aligned metric (precision/recall/MAE)",
    D: "Epoch loss"
  },
  correctAnswer: "C"
},
{
  id: 249,
  question: "Why F1-score is NOT always ideal?",
  options: {
    A: "Hard to compute",
    B: "Ignores true negatives and business costs",
    C: "Needs probabilities",
    D: "Only for regression"
  },
  correctAnswer: "B"
},
{
  id: 250,
  question: "Best evaluation strategy for real-world ML system:",
  options: {
    A: "Single metric only",
    B: "Accuracy only",
    C: "Multiple metrics + business context",
    D: "Training loss"
  },
  correctAnswer: "C"
},
{
  id: 251,
  question: "Missing values should be handled because they:",
  options: {
    A: "Always reduce accuracy",
    B: "Can break many ML algorithms",
    C: "Increase variance only",
    D: "Improve generalization"
  },
  correctAnswer: "B"
},
{
  id: 252,
  question: "Which method is suitable when missing values are very few?",
  options: {
    A: "Dropping rows/columns",
    B: "SMOTE",
    C: "Scaling",
    D: "Encoding"
  },
  correctAnswer: "A"
},
{
  id: 253,
  question: "Mean imputation is NOT recommended when:",
  options: {
    A: "Data is normally distributed",
    B: "Missing values are few",
    C: "Feature has strong outliers or skewness",
    D: "Feature is numerical"
  },
  correctAnswer: "C"
},
{
  id: 254,
  question: "Median imputation is preferred when:",
  options: {
    A: "Data has no outliers",
    B: "Data is categorical",
    C: "Data is skewed or has outliers",
    D: "Dataset is large"
  },
  correctAnswer: "C"
},
{
  id: 255,
  question: "Which algorithm can handle missing values natively?",
  options: {
    A: "Linear Regression",
    B: "KNN",
    C: "Decision Trees / XGBoost",
    D: "Logistic Regression"
  },
  correctAnswer: "C"
},
{
  id: 256,
  question: "Dropping a feature due to missing values is risky when:",
  options: {
    A: "Feature is irrelevant",
    B: "Feature has high predictive power",
    C: "Missing rate is high",
    D: "Dataset is small"
  },
  correctAnswer: "B"
},
{
  id: 257,
  question: "Outliers are problematic because they:",
  options: {
    A: "Increase dataset size",
    B: "Can distort model learning and metrics",
    C: "Always improve accuracy",
    D: "Only affect clustering"
  },
  correctAnswer: "B"
},
{
  id: 258,
  question: "Which method is commonly used to detect outliers?",
  options: {
    A: "Label encoding",
    B: "Z-score / IQR",
    C: "One-hot encoding",
    D: "PCA"
  },
  correctAnswer: "B"
},
{
  id: 259,
  question: "Z-score method assumes data is:",
  options: {
    A: "Uniform",
    B: "Skewed",
    C: "Normally distributed",
    D: "Categorical"
  },
  correctAnswer: "C"
},
{
  id: 260,
  question: "IQR method is preferred when:",
  options: {
    A: "Data is normally distributed",
    B: "Data has skewness and outliers",
    C: "Dataset is small",
    D: "Features are categorical"
  },
  correctAnswer: "B"
},
{
  id: 261,
  question: "Removing outliers is NOT recommended when:",
  options: {
    A: "They are due to data entry errors",
    B: "They represent real rare events (e.g., fraud)",
    C: "Dataset is large",
    D: "Model is linear"
  },
  correctAnswer: "B"
},
{
  id: 262,
  question: "Which model is most sensitive to outliers?",
  options: {
    A: "Decision Tree",
    B: "Random Forest",
    C: "Linear Regression",
    D: "DBSCAN"
  },
  correctAnswer: "C"
},
{
  id: 263,
  question: "Label Encoding is suitable when:",
  options: {
    A: "Categories have no order",
    B: "Model is tree-based and order does not mislead",
    C: "Data is numerical",
    D: "Features are binary"
  },
  correctAnswer: "B"
},
{
  id: 264,
  question: "One-Hot Encoding is preferred when:",
  options: {
    A: "Categories are ordinal",
    B: "Categories are nominal with no order",
    C: "Dataset is extremely large",
    D: "Feature has many unique values"
  },
  correctAnswer: "B"
},
{
  id: 265,
  question: "One-hot encoding can cause:",
  options: {
    A: "Data leakage",
    B: "Curse of dimensionality",
    C: "Overfitting always",
    D: "Feature scaling issues"
  },
  correctAnswer: "B"
},
{
  id: 266,
  question: "Target encoding is risky because it can:",
  options: {
    A: "Increase bias",
    B: "Cause data leakage if not done carefully",
    C: "Reduce dimensionality",
    D: "Improve generalization always"
  },
  correctAnswer: "B"
},
{
  id: 267,
  question: "High-cardinality categorical features are best handled using:",
  options: {
    A: "One-hot encoding",
    B: "Label encoding only",
    C: "Target encoding / embeddings",
    D: "Dropping the feature"
  },
  correctAnswer: "C"
},
{
  id: 268,
  question: "Feature scaling is important because:",
  options: {
    A: "It improves data quality",
    B: "Some models are sensitive to feature magnitude",
    C: "It removes outliers",
    D: "It encodes categories"
  },
  correctAnswer: "B"
},
{
  id: 269,
  question: "Which models require feature scaling?",
  options: {
    A: "Decision Trees",
    B: "Random Forest",
    C: "KNN, SVM, Linear Models",
    D: "Naive Bayes only"
  },
  correctAnswer: "C"
},
{
  id: 270,
  question: "StandardScaler transforms data to:",
  options: {
    A: "Range [0,1]",
    B: "Mean = 0, Std = 1",
    C: "Median = 0",
    D: "Unit length vectors"
  },
  correctAnswer: "B"
},
{
  id: 271,
  question: "MinMaxScaler is sensitive to:",
  options: {
    A: "Missing values",
    B: "Feature correlation",
    C: "Outliers",
    D: "Categorical data"
  },
  correctAnswer: "C"
},
{
  id: 272,
  question: "StandardScaler is preferred when:",
  options: {
    A: "Data has extreme outliers",
    B: "Algorithm assumes Gaussian distribution",
    C: "Features are bounded",
    D: "Dataset is very small"
  },
  correctAnswer: "B"
},
{
  id: 273,
  question: "Feature scaling should be done:",
  options: {
    A: "Before train-test split",
    B: "Using full dataset",
    C: "After split using training data only",
    D: "Only on test data"
  },
  correctAnswer: "C"
},
{
  id: 274,
  question: "An imbalanced dataset means:",
  options: {
    A: "Features are correlated",
    B: "One class dominates others in frequency",
    C: "Data has missing values",
    D: "Dataset is small"
  },
  correctAnswer: "B"
},
{
  id: 275,
  question: "Accuracy is misleading in imbalanced datasets because:",
  options: {
    A: "It ignores false positives",
    B: "Majority class dominates metric",
    C: "Model is biased",
    D: "Precision is low"
  },
  correctAnswer: "B"
},
{
  id: 276,
  question: "SMOTE works by:",
  options: {
    A: "Removing majority samples",
    B: "Creating synthetic minority samples",
    C: "Duplicating data",
    D: "Scaling features"
  },
  correctAnswer: "B"
},
{
  id: 277,
  question: "A risk of using SMOTE is:",
  options: {
    A: "Data leakage if applied before split",
    B: "Underfitting",
    C: "Reduced dataset size",
    D: "Loss of information"
  },
  correctAnswer: "A"
},
{
  id: 278,
  question: "Class weighting helps by:",
  options: {
    A: "Increasing dataset size",
    B: "Penalizing mistakes on minority class more",
    C: "Removing outliers",
    D: "Balancing features"
  },
  correctAnswer: "B"
},
{
  id: 279,
  question: "When class weights are preferred over SMOTE?",
  options: {
    A: "Dataset is very small",
    B: "Risk of synthetic data causing noise exists",
    C: "Data is balanced",
    D: "Features are categorical"
  },
  correctAnswer: "B"
},
{
  id: 280,
  question: "SMOTE should be applied:",
  options: {
    A: "Before splitting data",
    B: "On test set",
    C: "Only on training data",
    D: "On full dataset"
  },
  correctAnswer: "C"
},
{
  id: 281,
  question: "Preprocessing mismatch between training and inference causes:",
  options: {
    A: "Faster predictions",
    B: "Model drift",
    C: "Incorrect predictions in production",
    D: "Better accuracy"
  },
  correctAnswer: "C"
},
{
  id: 282,
  question: "Which preprocessing steps must be saved for inference?",
  options: {
    A: "Raw data",
    B: "Labels",
    C: "Scalers, encoders, imputers",
    D: "Metrics"
  },
  correctAnswer: "C"
},
{
  id: 283,
  question: "Why preprocessing pipelines are important?",
  options: {
    A: "For faster training",
    B: "For consistency and reproducibility",
    C: "For visualization",
    D: "For feature selection"
  },
  correctAnswer: "B"
},
{
  id: 284,
  question: "Which is a common production mistake?",
  options: {
    A: "Scaling training data",
    B: "Using same scaler for inference",
    C: "Re-fitting scaler on inference data",
    D: "Using pipelines"
  },
  correctAnswer: "C"
},
{
  id: 285,
  question: "Inference-time data may differ because of:",
  options: {
    A: "Label availability",
    B: "Data drift or missing features",
    C: "Scaling",
    D: "Encoding"
  },
  correctAnswer: "B"
},
{
  id: 286,
  question: "Model performance drops after deployment. First thing to check:",
  options: {
    A: "Algorithm",
    B: "Feature scaling mismatch",
    C: "Learning rate",
    D: "Epochs"
  },
  correctAnswer: "B"
},
{
  id: 287,
  question: "Dataset has 95% normal transactions, 5% fraud. Best approach:",
  options: {
    A: "Accuracy metric",
    B: "SMOTE or class weights + recall focus",
    C: "Remove fraud cases",
    D: "MinMax scaling"
  },
  correctAnswer: "B"
},
{
  id: 288,
  question: "Categorical feature with 1,000 unique values. Best encoding:",
  options: {
    A: "One-hot encoding",
    B: "Label encoding",
    C: "Target encoding / embeddings",
    D: "Drop feature"
  },
  correctAnswer: "C"
},
{
  id: 289,
  question: "Linear model performing poorly due to different feature ranges. Solution:",
  options: {
    A: "Drop features",
    B: "Feature scaling",
    C: "SMOTE",
    D: "PCA"
  },
  correctAnswer: "B"
},
{
  id: 290,
  question: "Outliers represent genuine rare events. What to do?",
  options: {
    A: "Remove them",
    B: "Clip values",
    C: "Keep and use robust models/metrics",
    D: "Ignore"
  },
  correctAnswer: "C"
},
{
  id: 291,
  question: "Which preprocessing step can cause data leakage most easily?",
  options: {
    A: "Scaling",
    B: "Encoding",
    C: "Target encoding & SMOTE before split",
    D: "Dropping columns"
  },
  correctAnswer: "C"
},
{
  id: 292,
  question: "Which models are least affected by feature scaling?",
  options: {
    A: "KNN",
    B: "SVM",
    C: "Decision Trees / Random Forest",
    D: "Linear Regression"
  },
  correctAnswer: "C"
},
{
  id: 293,
  question: "Missing values in target variable should be:",
  options: {
    A: "Imputed",
    B: "Dropped",
    C: "Filled with mean",
    D: "Label encoded"
  },
  correctAnswer: "B"
},
{
  id: 294,
  question: "Best way to ensure train–inference parity:",
  options: {
    A: "Manual preprocessing",
    B: "Documentation",
    C: "Automated preprocessing pipelines",
    D: "Feature scaling only"
  },
  correctAnswer: "C"
},
{
  id: 295,
  question: "Using future information to fill missing values causes:",
  options: {
    A: "Underfitting",
    B: "Bias",
    C: "Data leakage",
    D: "Noise"
  },
  correctAnswer: "C"
},
{
  id: 296,
  question: "Why MinMaxScaler may fail in production?",
  options: {
    A: "Slow",
    B: "Sensitive to new unseen extreme values",
    C: "Requires labels",
    D: "Needs retraining"
  },
  correctAnswer: "B"
},
{
  id: 297,
  question: "Which step should NEVER use test data?",
  options: {
    A: "Scaling fit",
    B: "Encoding fit",
    C: "SMOTE",
    D: "All of the above"
  },
  correctAnswer: "D"
},
{
  id: 298,
  question: "Model trained with StandardScaler, inference uses raw data. Result:",
  options: {
    A: "Better accuracy",
    B: "No change",
    C: "Incorrect predictions",
    D: "Faster inference"
  },
  correctAnswer: "C"
},
{
  id: 299,
  question: "Which preprocessing step depends on target variable?",
  options: {
    A: "One-hot encoding",
    B: "Scaling",
    C: "Target encoding",
    D: "PCA"
  },
  correctAnswer: "C"
},
{
  id: 300,
  question: "A production-ready preprocessing setup should be:",
  options: {
    A: "Manual scripts",
    B: "Notebook-based",
    C: "Pipeline-based & reusable",
    D: "Algorithm-specific"
  },
  correctAnswer: "C"
},
{
  id: 301,
  question: "Vectorization in NumPy means:",
  options: {
    A: "Writing loops in Python",
    B: "Using compiled operations instead of Python loops",
    C: "Converting arrays to vectors",
    D: "Using recursion"
  },
  correctAnswer: "B"
},
{
  id: 302,
  question: "Why is vectorized NumPy code faster?",
  options: {
    A: "Uses GPU automatically",
    B: "Runs in parallel C-code under the hood",
    C: "Uses caching",
    D: "Avoids memory"
  },
  correctAnswer: "B"
},
{
  id: 303,
  question: "Which operation benefits most from vectorization?",
  options: {
    A: "File I/O",
    B: "Element-wise array operations",
    C: "Conditional branching",
    D: "String processing"
  },
  correctAnswer: "B"
},
{
  id: 304,
  question: "Replacing Python loops with NumPy operations mainly improves:",
  options: {
    A: "Accuracy",
    B: "Readability only",
    C: "Performance & scalability",
    D: "Memory usage always"
  },
  correctAnswer: "C"
},
{
  id: 305,
  question: "Broadcasting allows NumPy to:",
  options: {
    A: "Share arrays across processes",
    B: "Operate on arrays of different shapes automatically",
    C: "Reduce memory",
    D: "Encode categories"
  },
  correctAnswer: "B"
},
{
  id: 306,
  question: "Pandas groupby is mainly used for:",
  options: {
    A: "Sorting data",
    B: "Aggregating data by keys",
    C: "Encoding features",
    D: "Scaling values"
  },
  correctAnswer: "B"
},
{
  id: 307,
  question: "Which operation is commonly used after groupby?",
  options: {
    A: "merge",
    B: "apply",
    C: "aggregation (sum, mean, count)",
    D: "join"
  },
  correctAnswer: "C"
},
{
  id: 308,
  question: "groupby helps reduce data by:",
  options: {
    A: "Filtering rows",
    B: "Aggregating multiple rows into summary statistics",
    C: "Dropping duplicates",
    D: "Encoding categories"
  },
  correctAnswer: "B"
},
{
  id: 309,
  question: "A common pitfall of groupby is:",
  options: {
    A: "High memory usage",
    B: "Loss of row-level detail",
    C: "Incorrect sorting",
    D: "Data leakage"
  },
  correctAnswer: "B"
},
{
  id: 310,
  question: "groupby is best suited for:",
  options: {
    A: "Row-wise operations",
    B: "Column-wise operations",
    C: "Split–apply–combine pattern",
    D: "Encoding"
  },
  correctAnswer: "C"
},
{
  id: 311,
  question: "merge in pandas is conceptually similar to:",
  options: {
    A: "Concatenation",
    B: "SQL JOIN",
    C: "groupby",
    D: "pivot"
  },
  correctAnswer: "B"
},
{
  id: 312,
  question: "Which join keeps only matching keys from both tables?",
  options: {
    A: "Left join",
    B: "Right join",
    C: "Inner join",
    D: "Outer join"
  },
  correctAnswer: "C"
},
{
  id: 313,
  question: "Left join ensures:",
  options: {
    A: "All rows from right table are kept",
    B: "All rows from left table are kept",
    C: "Only common rows are kept",
    D: "Rows are sorted"
  },
  correctAnswer: "B"
},
{
  id: 314,
  question: "A common issue during merge is:",
  options: {
    A: "Data leakage",
    B: "Duplicate rows due to many-to-many joins",
    C: "Feature scaling",
    D: "Missing labels"
  },
  correctAnswer: "B"
},
{
  id: 315,
  question: "After merging datasets, you should always check:",
  options: {
    A: "Accuracy",
    B: "Row count & duplicates",
    C: "Feature scaling",
    D: "Target balance"
  },
  correctAnswer: "B"
},
{
  id: 316,
  question: "DataFrame.apply() is:",
  options: {
    A: "Always faster than vectorized ops",
    B: "Python-level row/column iteration",
    C: "Compiled C-code",
    D: "GPU-accelerated"
  },
  correctAnswer: "B"
},
{
  id: 317,
  question: "Why should apply() be avoided on large datasets?",
  options: {
    A: "It causes data leakage",
    B: "It is slower than vectorized operations",
    C: "It increases bias",
    D: "It changes data"
  },
  correctAnswer: "B"
},
{
  id: 318,
  question: "Vectorized operations are preferred because they:",
  options: {
    A: "Are easier to write",
    B: "Scale better for large data",
    C: "Use more memory",
    D: "Are less readable"
  },
  correctAnswer: "B"
},
{
  id: 319,
  question: "When is apply() acceptable?",
  options: {
    A: "Always",
    B: "For small datasets or complex row-wise logic",
    C: "For performance-critical pipelines",
    D: "For encoding"
  },
  correctAnswer: "B"
},
{
  id: 320,
  question: "Best alternative to apply() for conditional logic is:",
  options: {
    A: "for-loop",
    B: "list comprehension",
    C: "NumPy boolean masking / vectorized ops",
    D: "recursion"
  },
  correctAnswer: "C"
},
{
  id: 321,
  question: "Clean ML code should prioritize:",
  options: {
    A: "Short scripts",
    B: "Copy-paste reuse",
    C: "Modularity & readability",
    D: "Notebook-only execution"
  },
  correctAnswer: "C"
},
{
  id: 322,
  question: "Which practice improves ML code maintainability?",
  options: {
    A: "Hardcoding paths",
    B: "Global variables",
    C: "Functions & classes for pipelines",
    D: "Inline logic everywhere"
  },
  correctAnswer: "C"
},
{
  id: 323,
  question: "Configuration (paths, params) should be:",
  options: {
    A: "Hardcoded",
    B: "Stored inside functions",
    C: "Externalized (config files / env vars)",
    D: "Ignored"
  },
  correctAnswer: "C"
},
{
  id: 324,
  question: "Why should data preprocessing be reusable?",
  options: {
    A: "To save memory",
    B: "For consistent training & inference",
    C: "To reduce features",
    D: "To improve accuracy automatically"
  },
  correctAnswer: "B"
},
{
  id: 325,
  question: "Which structure is BEST for ML projects?",
  options: {
    A: "Single notebook",
    B: "Scripts with no folders",
    C: "Modular folders (data, features, models, utils)",
    D: "One huge file"
  },
  correctAnswer: "C"
},
{
  id: 326,
  question: "When data doesn’t fit in memory, use:",
  options: {
    A: "list",
    B: "NumPy only",
    C: "Chunking / streaming (read in chunks)",
    D: "apply()"
  },
  correctAnswer: "C"
},
{
  id: 327,
  question: "Reading CSV in chunks helps because:",
  options: {
    A: "It increases speed always",
    B: "It reduces memory usage",
    C: "It avoids preprocessing",
    D: "It improves accuracy"
  },
  correctAnswer: "B"
},
{
  id: 328,
  question: "Which datatype choice helps reduce memory usage?",
  options: {
    A: "float64 everywhere",
    B: "object dtype",
    C: "Appropriate dtypes (int32, category)",
    D: "strings"
  },
  correctAnswer: "C"
},
{
  id: 329,
  question: "For large numerical datasets, which is faster?",
  options: {
    A: "Python lists",
    B: "Dictionaries",
    C: "NumPy arrays / Pandas vectors",
    D: "JSON"
  },
  correctAnswer: "C"
},
{
  id: 330,
  question: "Which operation is expensive on large DataFrames?",
  options: {
    A: "Vectorized math",
    B: "Boolean indexing",
    C: "Row-wise apply()",
    D: "Aggregation"
  },
  correctAnswer: "C"
},
{
  id: 331,
  question: "Why use OOP in ML pipelines?",
  options: {
    A: "Mandatory for ML",
    B: "For code organization & reusability",
    C: "Faster execution",
    D: "Better accuracy"
  },
  correctAnswer: "B"
},
{
  id: 332,
  question: "A typical ML pipeline class should encapsulate:",
  options: {
    A: "Only model",
    B: "Only data loading",
    C: "Preprocessing + training + inference logic",
    D: "Metrics only"
  },
  correctAnswer: "C"
},
{
  id: 333,
  question: "Which principle helps extend ML pipelines easily?",
  options: {
    A: "Hardcoding",
    B: "Copy-paste",
    C: "Separation of concerns",
    D: "Global state"
  },
  correctAnswer: "C"
},
{
  id: 334,
  question: "In ML code, inheritance is useful when:",
  options: {
    A: "Writing scripts",
    B: "Sharing common behavior across models",
    C: "Handling CSV files",
    D: "Scaling features"
  },
  correctAnswer: "B"
},
{
  id: 335,
  question: "Composition over inheritance means:",
  options: {
    A: "Avoid classes",
    B: "Use objects inside classes for flexibility",
    C: "Write longer code",
    D: "Inherit everything"
  },
  correctAnswer: "B"
},
{
  id: 336,
  question: "Model training is slow due to Python loops. Best fix:",
  options: {
    A: "Use apply()",
    B: "Rewrite logic using NumPy vectorization",
    C: "Increase RAM",
    D: "Add print logs"
  },
  correctAnswer: "B"
},
{
  id: 337,
  question: "After merging two datasets, row count doubled unexpectedly. Cause:",
  options: {
    A: "Feature scaling",
    B: "Many-to-many join keys",
    C: "Missing values",
    D: "Pandas bug"
  },
  correctAnswer: "B"
},
{
  id: 338,
  question: "Model predictions differ between notebook and production. First check:",
  options: {
    A: "Python version",
    B: "Preprocessing consistency & pipeline reuse",
    C: "Algorithm",
    D: "GPU usage"
  },
  correctAnswer: "B"
},
{
  id: 339,
  question: "You need same preprocessing during training and inference. Best approach:",
  options: {
    A: "Duplicate code",
    B: "Copy notebook",
    C: "Create reusable preprocessing class/function",
    D: "Inline logic"
  },
  correctAnswer: "C"
},
{
  id: 340,
  question: "Pandas code is slow on millions of rows. Best approach:",
  options: {
    A: "apply()",
    B: "Python loops",
    C: "Vectorization / chunking / dtype optimization",
    D: "Print statements"
  },
  correctAnswer: "C"
},
{
  id: 341,
  question: "Which pandas operation is lazy (delayed execution)?",
  options: {
    A: "apply",
    B: "merge",
    C: "None (pandas is eager)",
    D: "groupby"
  },
  correctAnswer: "C"
},
{
  id: 342,
  question: "Why avoid global variables in ML code?",
  options: {
    A: "Slower performance",
    B: "Harder debugging & reproducibility issues",
    C: "More memory usage",
    D: "Syntax errors"
  },
  correctAnswer: "B"
},
{
  id: 343,
  question: "A pipeline class improves:",
  options: {
    A: "Accuracy",
    B: "Reproducibility & deployment readiness",
    C: "Dataset size",
    D: "Label quality"
  },
  correctAnswer: "B"
},
{
  id: 344,
  question: "For feature engineering logic reused across projects, use:",
  options: {
    A: "Notebook cells",
    B: "Inline scripts",
    C: "Utility modules / packages",
    D: "Comments"
  },
  correctAnswer: "C"
},
{
  id: 345,
  question: "Which Python feature helps manage large loops efficiently?",
  options: {
    A: "Recursion",
    B: "Generators & iterators",
    C: "Deep copy",
    D: "Globals"
  },
  correctAnswer: "B"
},
{
  id: 346,
  question: "Best way to test ML utility functions:",
  options: {
    A: "Manual testing",
    B: "Print statements",
    C: "Unit tests (pytest)",
    D: "Logging only"
  },
  correctAnswer: "C"
},
{
  id: 347,
  question: "Pandas category dtype is useful because:",
  options: {
    A: "Faster math",
    B: "Reduces memory for categorical features",
    C: "Improves accuracy",
    D: "Encodes labels"
  },
  correctAnswer: "B"
},
{
  id: 348,
  question: "Which practice helps reproducibility?",
  options: {
    A: "Random code execution",
    B: "Fixing random seeds & versions",
    C: "Using apply()",
    D: "Hardcoding values"
  },
  correctAnswer: "B"
},
{
  id: 349,
  question: "Production ML code should be:",
  options: {
    A: "Notebook-driven",
    B: "Script-only",
    C: "Modular, tested, and reusable",
    D: "Quick hacks"
  },
  correctAnswer: "C"
},
{
  id: 350,
  question: "A strong Python-for-AI engineer prioritizes:",
  options: {
    A: "Clever one-liners",
    B: "Syntax tricks",
    C: "Clarity, performance, and correctness",
    D: "Notebook visuals"
  },
  correctAnswer: "C"
},
{
  id: 351,
  question: "A neural network layer consists of:",
  options: {
    A: "Only neurons",
    B: "Weights, bias, and activation function",
    C: "Loss function",
    D: "Optimizer"
  },
  correctAnswer: "B"
},
{
  id: 352,
  question: "Weights in a neural network represent:",
  options: {
    A: "Output labels",
    B: "Feature importance learned from data",
    C: "Learning rate",
    D: "Noise"
  },
  correctAnswer: "B"
},
{
  id: 353,
  question: "Bias term is used to:",
  options: {
    A: "Increase variance",
    B: "Shift activation function output",
    C: "Reduce features",
    D: "Normalize data"
  },
  correctAnswer: "B"
},
{
  id: 354,
  question: "Increasing number of layers generally:",
  options: {
    A: "Reduces model capacity",
    B: "Increases ability to learn complex patterns",
    C: "Always improves performance",
    D: "Removes overfitting"
  },
  correctAnswer: "B"
},
{
  id: 355,
  question: "Neural networks are powerful because they:",
  options: {
    A: "Use probabilities",
    B: "Learn hierarchical representations",
    C: "Require no data preprocessing",
    D: "Avoid overfitting"
  },
  correctAnswer: "B"
},
{
  id: 356,
  question: "ReLU activation outputs:",
  options: {
    A: "Values between 0 and 1",
    B: "Values between -1 and 1",
    C: "max(0, x)",
    D: "Only binary values"
  },
  correctAnswer: "C"
},
{
  id: 357,
  question: "Why is ReLU widely used in hidden layers?",
  options: {
    A: "Prevents overfitting",
    B: "Avoids vanishing gradient to some extent",
    C: "Produces probabilities",
    D: "Works only for classification"
  },
  correctAnswer: "B"
},
{
  id: 358,
  question: "Sigmoid activation is mainly used in:",
  options: {
    A: "Hidden layers",
    B: "Multi-class output",
    C: "Binary classification output layer",
    D: "Regression"
  },
  correctAnswer: "C"
},
{
  id: 359,
  question: "Softmax activation is used when:",
  options: {
    A: "Predicting continuous values",
    B: "Multi-class classification with probabilities",
    C: "Binary classification",
    D: "Feature scaling"
  },
  correctAnswer: "B"
},
{
  id: 360,
  question: "A drawback of Sigmoid activation is:",
  options: {
    A: "Non-linearity",
    B: "Saturation causing vanishing gradients",
    C: "Slow inference",
    D: "Produces negative values"
  },
  correctAnswer: "B"
},
{
  id: 361,
  question: "Loss function measures:",
  options: {
    A: "Model accuracy",
    B: "Difference between predicted and actual output",
    C: "Learning rate",
    D: "Feature importance"
  },
  correctAnswer: "B"
},
{
  id: 362,
  question: "Which loss is commonly used for regression?",
  options: {
    A: "Cross-entropy",
    B: "Hinge loss",
    C: "Mean Squared Error (MSE)",
    D: "Log loss"
  },
  correctAnswer: "C"
},
{
  id: 363,
  question: "Binary cross-entropy is used when:",
  options: {
    A: "Output is continuous",
    B: "Binary classification problem",
    C: "Multi-class classification",
    D: "Clustering"
  },
  correctAnswer: "B"
},
{
  id: 364,
  question: "Categorical cross-entropy is used for:",
  options: {
    A: "Regression",
    B: "Binary classification",
    C: "Multi-class classification with softmax",
    D: "Autoencoders"
  },
  correctAnswer: "C"
},
{
  id: 365,
  question: "Choosing wrong loss function can lead to:",
  options: {
    A: "Faster training",
    B: "Poor convergence or incorrect learning",
    C: "Better generalization",
    D: "Less overfitting"
  },
  correctAnswer: "B"
},
{
  id: 366,
  question: "Backpropagation is used to:",
  options: {
    A: "Initialize weights",
    B: "Compute gradients for updating weights",
    C: "Reduce dataset size",
    D: "Choose activation functions"
  },
  correctAnswer: "B"
},
{
  id: 367,
  question: "Gradients indicate:",
  options: {
    A: "Direction and magnitude of weight updates",
    B: "Final predictions",
    C: "Model accuracy",
    D: "Bias only"
  },
  correctAnswer: "A"
},
{
  id: 368,
  question: "Vanishing gradient problem occurs when:",
  options: {
    A: "Gradients become too large",
    B: "Gradients become very small in deep networks",
    C: "Learning rate is high",
    D: "Batch size is large"
  },
  correctAnswer: "B"
},
{
  id: 369,
  question: "ReLU helps mitigate vanishing gradients because:",
  options: {
    A: "It is linear everywhere",
    B: "It does not saturate for positive values",
    C: "It outputs probabilities",
    D: "It normalizes data"
  },
  correctAnswer: "B"
},
{
  id: 370,
  question: "Backpropagation combined with gradient descent enables:",
  options: {
    A: "Random learning",
    B: "End-to-end training of neural networks",
    C: "Feature selection only",
    D: "Clustering"
  },
  correctAnswer: "B"
},
{
  id: 371,
  question: "Overfitting in deep learning happens because:",
  options: {
    A: "Model is too simple",
    B: "Model has many parameters and memorizes data",
    C: "Data is always noisy",
    D: "Loss function is wrong"
  },
  correctAnswer: "B"
},
{
  id: 372,
  question: "Which technique helps reduce overfitting?",
  options: {
    A: "Increasing layers blindly",
    B: "Dropout",
    C: "Increasing batch size",
    D: "Removing validation set"
  },
  correctAnswer: "B"
},
{
  id: 373,
  question: "Dropout works by:",
  options: {
    A: "Removing neurons permanently",
    B: "Randomly disabling neurons during training",
    C: "Scaling weights",
    D: "Reducing learning rate"
  },
  correctAnswer: "B"
},
{
  id: 374,
  question: "L2 regularization helps by:",
  options: {
    A: "Increasing weights",
    B: "Penalizing large weights",
    C: "Removing features",
    D: "Adding noise"
  },
  correctAnswer: "B"
},
{
  id: 375,
  question: "Early stopping prevents overfitting by:",
  options: {
    A: "Stopping training when validation loss worsens",
    B: "Increasing epochs",
    C: "Increasing learning rate",
    D: "Adding layers"
  },
  correctAnswer: "A"
},
{
  id: 376,
  question: "Data augmentation helps by:",
  options: {
    A: "Reducing dataset size",
    B: "Increasing effective training data variability",
    C: "Removing noise",
    D: "Scaling features"
  },
  correctAnswer: "B"
},
{
  id: 377,
  question: "CNNs are best suited for:",
  options: {
    A: "Time-series forecasting",
    B: "Image and spatial data processing",
    C: "Text generation",
    D: "Tabular data only"
  },
  correctAnswer: "B"
},
{
  id: 378,
  question: "CNNs exploit:",
  options: {
    A: "Sequential dependency",
    B: "Spatial locality and parameter sharing",
    C: "Attention mechanism",
    D: "Recurrence"
  },
  correctAnswer: "B"
},
{
  id: 379,
  question: "RNNs are designed to handle:",
  options: {
    A: "Images",
    B: "Independent samples",
    C: "Sequential data with temporal dependency",
    D: "Clustering"
  },
  correctAnswer: "C"
},
{
  id: 380,
  question: "A limitation of vanilla RNNs is:",
  options: {
    A: "Too many parameters",
    B: "Difficulty learning long-term dependencies",
    C: "No activation functions",
    D: "High interpretability"
  },
  correctAnswer: "B"
},
{
  id: 381,
  question: "LSTM and GRU were introduced to:",
  options: {
    A: "Increase parameters",
    B: "Solve vanishing gradient in sequences",
    C: "Speed up CNNs",
    D: "Replace transformers"
  },
  correctAnswer: "B"
},
{
  id: 382,
  question: "Transformers differ from RNNs because they:",
  options: {
    A: "Process data sequentially",
    B: "Use attention instead of recurrence",
    C: "Cannot handle text",
    D: "Are slower always"
  },
  correctAnswer: "B"
},
{
  id: 383,
  question: "Self-attention allows models to:",
  options: {
    A: "Ignore context",
    B: "Focus on relevant parts of input sequence",
    C: "Reduce dataset size",
    D: "Encode labels"
  },
  correctAnswer: "B"
},
{
  id: 384,
  question: "Transformers are preferred when:",
  options: {
    A: "Data is very small",
    B: "Long-range dependencies matter and parallelization is needed",
    C: "Interpretability is key",
    D: "Only images are involved"
  },
  correctAnswer: "B"
},
{
  id: 385,
  question: "A drawback of Transformers is:",
  options: {
    A: "Poor performance",
    B: "High computational and memory cost",
    C: "Inability to model context",
    D: "Overfitting always"
  },
  correctAnswer: "B"
},
{
  id: 386,
  question: "You are building an image classifier. Best architecture:",
  options: {
    A: "RNN",
    B: "Transformer only",
    C: "CNN",
    D: "Logistic Regression"
  },
  correctAnswer: "C"
},
{
  id: 387,
  question: "Predicting next word in a sentence. Best choice:",
  options: {
    A: "CNN",
    B: "RNN / Transformer",
    C: "K-Means",
    D: "PCA"
  },
  correctAnswer: "B"
},
{
  id: 388,
  question: "Processing long documents with global context. Best model:",
  options: {
    A: "RNN",
    B: "CNN",
    C: "Transformer",
    D: "Naive Bayes"
  },
  correctAnswer: "C"
},
{
  id: 389,
  question: "Model trains well but validation loss increases. Issue:",
  options: {
    A: "Underfitting",
    B: "Overfitting",
    C: "Vanishing gradient",
    D: "Exploding gradient"
  },
  correctAnswer: "B"
},
{
  id: 390,
  question: "Which technique helps most when data is limited?",
  options: {
    A: "Larger model",
    B: "Data augmentation & transfer learning",
    C: "Removing regularization",
    D: "Increasing epochs"
  },
  correctAnswer: "B"
},
{
  id: 391,
  question: "Which activation is NOT suitable for output layer of regression?",
  options: {
    A: "Linear",
    B: "ReLU",
    C: "Softmax",
    D: "Identity"
  },
  correctAnswer: "C"
},
{
  id: 392,
  question: "Softmax output values:",
  options: {
    A: "Are independent",
    B: "Sum to 1 (probability distribution)",
    C: "Can be negative",
    D: "Are unbounded"
  },
  correctAnswer: "B"
},
{
  id: 393,
  question: "Training very deep networks became feasible mainly due to:",
  options: {
    A: "Bigger datasets",
    B: "ReLU & better optimization techniques",
    C: "More CPUs",
    D: "Dropout only"
  },
  correctAnswer: "B"
},
{
  id: 394,
  question: "Which regularization technique increases robustness to noise?",
  options: {
    A: "Dropout",
    B: "Data augmentation",
    C: "Early stopping",
    D: "Weight decay"
  },
  correctAnswer: "B"
},
{
  id: 395,
  question: "If model gradients explode, common fix is:",
  options: {
    A: "Increase learning rate",
    B: "Gradient clipping",
    C: "Remove regularization",
    D: "Increase batch size"
  },
  correctAnswer: "B"
},
{
  id: 396,
  question: "Which architecture handles variable-length input naturally?",
  options: {
    A: "CNN",
    B: "RNN / Transformer",
    C: "Linear Regression",
    D: "PCA"
  },
  correctAnswer: "B"
},
{
  id: 397,
  question: "Why CNNs use shared weights?",
  options: {
    A: "Increase parameters",
    B: "Reduce parameters and capture local patterns",
    C: "Improve interpretability",
    D: "Avoid preprocessing"
  },
  correctAnswer: "B"
},
{
  id: 398,
  question: "Which DL model is most parallelizable during training?",
  options: {
    A: "RNN",
    B: "LSTM",
    C: "Transformer",
    D: "CNN only"
  },
  correctAnswer: "C"
},
{
  id: 399,
  question: "For tabular data, deep learning often:",
  options: {
    A: "Always outperforms ML",
    B: "Is unnecessary compared to tree-based models",
    C: "Requires no preprocessing",
    D: "Needs CNN"
  },
  correctAnswer: "B"
},
{
  id: 400,
  question: "A mid-level DL engineer should focus on:",
  options: {
    A: "Writing backprop math",
    B: "Model architecture selection & regularization",
    C: "Research papers only",
    D: "Reinventing optimizers"
  },
  correctAnswer: "B"
},
{
  id: 401,
  question: "A complete ML pipeline includes:",
  options: {
    A: "Data → Model → Prediction",
    B: "Data collection → preprocessing → training → evaluation → deployment → monitoring",
    C: "Model → API → UI",
    D: "Data labeling only"
  },
  correctAnswer: "B"
},
{
  id: 402,
  question: "Which step is most often underestimated in real projects?",
  options: {
    A: "Model selection",
    B: "Data quality & preprocessing",
    C: "Algorithm choice",
    D: "GPU selection"
  },
  correctAnswer: "B"
},
{
  id: 403,
  question: "Feature engineering should be:",
  options: {
    A: "Done only in notebooks",
    B: "Consistent between training and inference",
    C: "Done after deployment",
    D: "Avoided in DL systems"
  },
  correctAnswer: "B"
},
{
  id: 404,
  question: "Why pipelines are preferred over scripts?",
  options: {
    A: "Faster coding",
    B: "Reproducibility & automation",
    C: "More accuracy",
    D: "Less cost"
  },
  correctAnswer: "B"
},
{
  id: 405,
  question: "Production ML systems fail mostly due to:",
  options: {
    A: "Bad models",
    B: "Data issues & system design flaws",
    C: "Poor GPUs",
    D: "LLM hallucinations"
  },
  correctAnswer: "B"
},
{
  id: 406,
  question: "Model training is usually:",
  options: {
    A: "Latency-sensitive",
    B: "Compute-heavy & offline",
    C: "Stateless",
    D: "Real-time"
  },
  correctAnswer: "B"
},
{
  id: 407,
  question: "Model serving is usually:",
  options: {
    A: "Offline",
    B: "Batch-only",
    C: "Latency-sensitive & real-time",
    D: "GPU-intensive always"
  },
  correctAnswer: "C"
},
{
  id: 408,
  question: "Why training and serving environments differ?",
  options: {
    A: "Different languages",
    B: "Different performance & resource needs",
    C: "Different data formats",
    D: "Different models"
  },
  correctAnswer: "B"
},
{
  id: 409,
  question: "A common serving mistake is:",
  options: {
    A: "Using REST",
    B: "Recomputing heavy features at request time",
    C: "Using Docker",
    D: "Logging predictions"
  },
  correctAnswer: "B"
},
{
  id: 410,
  question: "Model serving should prioritize:",
  options: {
    A: "Accuracy only",
    B: "Latency, reliability, and scalability",
    C: "GPU usage",
    D: "Retraining speed"
  },
  correctAnswer: "B"
},
{
  id: 411,
  question: "Batch inference is best when:",
  options: {
    A: "Predictions needed instantly",
    B: "Latency is critical",
    C: "Large volumes processed periodically",
    D: "User-facing apps"
  },
  correctAnswer: "C"
},
{
  id: 412,
  question: "Real-time inference is required for:",
  options: {
    A: "Monthly reports",
    B: "Offline analytics",
    C: "Fraud detection, recommendations",
    D: "Data labeling"
  },
  correctAnswer: "C"
},
{
  id: 413,
  question: "Batch inference typically optimizes for:",
  options: {
    A: "Latency",
    B: "Throughput & cost efficiency",
    C: "Accuracy only",
    D: "Memory"
  },
  correctAnswer: "B"
},
{
  id: 414,
  question: "Real-time inference optimizes for:",
  options: {
    A: "Throughput only",
    B: "Lowest latency possible",
    C: "Cost per request",
    D: "Storage"
  },
  correctAnswer: "B"
},
{
  id: 415,
  question: "Which system is more complex to operate?",
  options: {
    A: "Batch inference",
    B: "Real-time inference",
    C: "Offline training",
    D: "Feature engineering"
  },
  correctAnswer: "B"
},
{
  id: 416,
  question: "Increasing model size usually:",
  options: {
    A: "Reduces latency",
    B: "Improves accuracy but increases latency",
    C: "Reduces cost",
    D: "Improves reliability"
  },
  correctAnswer: "B"
},
{
  id: 417,
  question: "In user-facing systems, acceptable latency is often:",
  options: {
    A: "Minutes",
    B: "Seconds or less",
    C: "Hours",
    D: "Depends only on model"
  },
  correctAnswer: "B"
},
{
  id: 418,
  question: "A common way to reduce latency is:",
  options: {
    A: "Larger model",
    B: "Model quantization or distillation",
    C: "More features",
    D: "Higher temperature"
  },
  correctAnswer: "B"
},
{
  id: 419,
  question: "Caching predictions helps when:",
  options: {
    A: "Inputs are always unique",
    B: "Inputs repeat frequently",
    C: "Accuracy is low",
    D: "Training often"
  },
  correctAnswer: "B"
},
{
  id: 420,
  question: "Best tradeoff strategy in production:",
  options: {
    A: "Max accuracy always",
    B: "Balance accuracy, latency, and cost based on use case",
    C: "Lowest cost only",
    D: "Largest model"
  },
  correctAnswer: "B"
},
{
  id: 421,
  question: "Model versioning is needed to:",
  options: {
    A: "Improve accuracy",
    B: "Track, roll back, and reproduce models",
    C: "Reduce latency",
    D: "Store embeddings"
  },
  correctAnswer: "B"
},
{
  id: 422,
  question: "What should be versioned along with model?",
  options: {
    A: "Only weights",
    B: "Code, data schema, features, hyperparameters",
    C: "UI",
    D: "Logs only"
  },
  correctAnswer: "B"
},
{
  id: 423,
  question: "Canary deployment means:",
  options: {
    A: "Deploying to all users",
    B: "Testing new model on small traffic first",
    C: "Offline testing",
    D: "Rolling back immediately"
  },
  correctAnswer: "B"
},
{
  id: 424,
  question: "Blue-green deployment allows:",
  options: {
    A: "Parallel training",
    B: "Zero-downtime model switching",
    C: "Faster training",
    D: "More accuracy"
  },
  correctAnswer: "B"
},
{
  id: 425,
  question: "Without versioning, debugging production issues becomes:",
  options: {
    A: "Easier",
    B: "Nearly impossible",
    C: "Faster",
    D: "Automatic"
  },
  correctAnswer: "B"
},
{
  id: 426,
  question: "Data drift means:",
  options: {
    A: "Model parameters change",
    B: "Input data distribution changes over time",
    C: "Accuracy improves",
    D: "GPU usage spikes"
  },
  correctAnswer: "B"
},
{
  id: 427,
  question: "Concept drift means:",
  options: {
    A: "Input format changes",
    B: "Relationship between input and target changes",
    C: "Token limit exceeded",
    D: "Vector DB update"
  },
  correctAnswer: "B"
},
{
  id: 428,
  question: "Which metric helps detect drift?",
  options: {
    A: "Training loss",
    B: "Prediction distribution monitoring",
    C: "GPU utilization",
    D: "Batch size"
  },
  correctAnswer: "B"
},
{
  id: 429,
  question: "Why drift is dangerous?",
  options: {
    A: "Models crash",
    B: "Silent performance degradation",
    C: "Higher cost",
    D: "Faster inference"
  },
  correctAnswer: "B"
},
{
  id: 430,
  question: "Drift detection is harder when:",
  options: {
    A: "Labels are delayed or unavailable",
    B: "Data is structured",
    C: "Model is simple",
    D: "Batch inference"
  },
  correctAnswer: "A"
},
{
  id: 431,
  question: "Retraining should be triggered by:",
  options: {
    A: "Fixed time only",
    B: "Performance drop or drift detection",
    C: "User complaints only",
    D: "GPU availability"
  },
  correctAnswer: "B"
},
{
  id: 432,
  question: "Scheduled retraining works best when:",
  options: {
    A: "Data changes unpredictably",
    B: "Data patterns are stable and periodic",
    C: "Labels unavailable",
    D: "Real-time systems only"
  },
  correctAnswer: "B"
},
{
  id: 433,
  question: "Continuous retraining risks include:",
  options: {
    A: "Faster adaptation",
    B: "Training on noisy or biased data",
    C: "Lower latency",
    D: "Higher accuracy always"
  },
  correctAnswer: "B"
},
{
  id: 434,
  question: "Incremental learning helps by:",
  options: {
    A: "Retraining from scratch always",
    B: "Updating model with new data efficiently",
    C: "Removing old data",
    D: "Reducing storage"
  },
  correctAnswer: "B"
},
{
  id: 435,
  question: "Before retraining, you should:",
  options: {
    A: "Delete old model",
    B: "Analyze data quality & drift source",
    C: "Increase model size",
    D: "Change framework"
  },
  correctAnswer: "B"
},
{
  id: 436,
  question: "Your model accuracy drops slowly over months. Likely cause:",
  options: {
    A: "Bug",
    B: "Data drift",
    C: "Hardware failure",
    D: "Overfitting"
  },
  correctAnswer: "B"
},
{
  id: 437,
  question: "Users complain about slow predictions. Best first step:",
  options: {
    A: "Retrain model",
    B: "Profile serving latency & bottlenecks",
    C: "Increase accuracy",
    D: "Change algorithm"
  },
  correctAnswer: "B"
},
{
  id: 438,
  question: "A/B testing models helps to:",
  options: {
    A: "Train faster",
    B: "Compare performance on real traffic",
    C: "Reduce drift",
    D: "Improve embeddings"
  },
  correctAnswer: "B"
},
{
  id: 439,
  question: "Which system needs strongest monitoring?",
  options: {
    A: "Offline batch reports",
    B: "User-facing real-time AI systems",
    C: "Training pipelines",
    D: "Data labeling"
  },
  correctAnswer: "B"
},
{
  id: 440,
  question: "Model outputs change drastically after retraining. Reason:",
  options: {
    A: "Better model",
    B: "Data shift or leakage introduced",
    C: "GPU change",
    D: "Tokenization"
  },
  correctAnswer: "B"
},
{
  id: 441,
  question: "Why feature stores are used?",
  options: {
    A: "Store raw data",
    B: "Ensure feature consistency across training & serving",
    C: "Speed up GPUs",
    D: "Reduce hallucinations"
  },
  correctAnswer: "B"
},
{
  id: 442,
  question: "Shadow deployment means:",
  options: {
    A: "No deployment",
    B: "New model runs alongside old without affecting users",
    C: "Model rollback",
    D: "Offline testing"
  },
  correctAnswer: "B"
},
{
  id: 443,
  question: "Logging predictions is important to:",
  options: {
    A: "Improve UI",
    B: "Debug, audit, and retrain models",
    C: "Reduce latency",
    D: "Improve embeddings"
  },
  correctAnswer: "B"
},
{
  id: 444,
  question: "Which is NOT a system design concern?",
  options: {
    A: "Latency",
    B: "Scalability",
    C: "Interpretability",
    D: "Model architecture details only"
  },
  correctAnswer: "D"
},
{
  id: 445,
  question: "Real-time LLM apps often need:",
  options: {
    A: "High temperature",
    B: "Caching, rate limits, fallbacks",
    C: "Fine-tuning only",
    D: "Batch inference"
  },
  correctAnswer: "B"
},
{
  id: 446,
  question: "Cost in production ML is mostly driven by:",
  options: {
    A: "Model accuracy",
    B: "Inference frequency & infrastructure",
    C: "Feature count",
    D: "Loss function"
  },
  correctAnswer: "B"
},
{
  id: 447,
  question: "Why rollback capability is critical?",
  options: {
    A: "Improves training",
    B: "Quickly recover from faulty deployments",
    C: "Reduces drift",
    D: "Improves UX"
  },
  correctAnswer: "B"
},
{
  id: 448,
  question: "Which metric matters most in production?",
  options: {
    A: "Training accuracy",
    B: "Business KPI impact & live performance",
    C: "Validation loss",
    D: "GPU utilization"
  },
  correctAnswer: "B"
},
{
  id: 449,
  question: "A mature AI system is:",
  options: {
    A: "Model-centric",
    B: "System-centric with monitoring & controls",
    C: "Prompt-centric",
    D: "Research-focused"
  },
  correctAnswer: "B"
},
{
  id: 450,
  question: "Real AI engineers are identified by their ability to:",
  options: {
    A: "Train large models",
    B: "Design reliable, monitored, scalable AI systems",
    C: "Write papers",
    D: "Tune hyperparameters only"
  },
  correctAnswer: "B"
},
{
  id: 451,
  question: "A model performs well during training but fails in production. MOST likely reason?",
  options: {
    A: "Model architecture is wrong",
    B: "Training data distribution differs from production data",
    C: "Learning rate too low",
    D: "Model underfit"
  },
  correctAnswer: "B"
},
{
  id: 452,
  question: "Training accuracy is high, validation accuracy is good, but live predictions are poor. First thing to check?",
  options: {
    A: "Increase epochs",
    B: "Feature preprocessing mismatch between training & inference",
    C: "Add more layers",
    D: "Change loss function"
  },
  correctAnswer: "B"
},
{
  id: 453,
  question: "Model works in notebook but not after deployment. Common cause?",
  options: {
    A: "Model file corrupted",
    B: "Missing feature engineering logic in serving code",
    C: "GPU vs CPU difference",
    D: "Batch size mismatch"
  },
  correctAnswer: "B"
},
{
  id: 454,
  question: "Model predictions slowly degrade over months. Likely issue?",
  options: {
    A: "Bug in code",
    B: "Data drift in production inputs",
    C: "Overfitting",
    D: "Tokenization error"
  },
  correctAnswer: "B"
},
{
  id: 455,
  question: "Production model fails only for new users. Root cause?",
  options: {
    A: "Model size",
    B: "Training data lacked similar user profiles",
    C: "High latency",
    D: "GPU memory"
  },
  correctAnswer: "B"
},
{
  id: 456,
  question: "Model accuracy is 95%, but business revenue drops. What went wrong?",
  options: {
    A: "Model too slow",
    B: "Wrong evaluation metric used for business goal",
    C: "Overfitting",
    D: "Dataset too small"
  },
  correctAnswer: "B"
},
{
  id: 457,
  question: "A fraud detection model has high accuracy but misses most fraud cases. Why?",
  options: {
    A: "Overfitting",
    B: "Class imbalance + accuracy is misleading",
    C: "Feature scaling",
    D: "Wrong optimizer"
  },
  correctAnswer: "B"
},
{
  id: 458,
  question: "Model optimized for RMSE, but users complain. Reason?",
  options: {
    A: "RMSE is wrong metric",
    B: "Metric doesn’t align with user experience or business impact",
    C: "Model underfit",
    D: "Batch size small"
  },
  correctAnswer: "B"
},
{
  id: 459,
  question: "Recommendation model improves CTR but reduces long-term retention. Why?",
  options: {
    A: "Model too complex",
    B: "Optimizing short-term metric ignored long-term KPIs",
    C: "Feature leakage",
    D: "Cold start"
  },
  correctAnswer: "B"
},
{
  id: 460,
  question: "Business team says model is “correct but useless.” Meaning?",
  options: {
    A: "Low accuracy",
    B: "Model output not actionable or aligned with decisions",
    C: "Model slow",
    D: "Model unstable"
  },
  correctAnswer: "B"
},
{
  id: 461,
  question: "LLM gives fluent but wrong legal advice. BEST fix?",
  options: {
    A: "Increase temperature",
    B: "Add RAG with authoritative legal sources",
    C: "Fine-tune on random data",
    D: "Use larger model only"
  },
  correctAnswer: "B"
},
{
  id: 462,
  question: "LLM answers differently every run. How to stabilize?",
  options: {
    A: "Increase top-p",
    B: "Reduce temperature & control prompts",
    C: "Increase context window",
    D: "Add emojis"
  },
  correctAnswer: "B"
},
{
  id: 463,
  question: "LLM invents facts when documents are missing. Why?",
  options: {
    A: "Bug",
    B: "LLM tries to be helpful even without grounding",
    C: "Token overflow",
    D: "Model size"
  },
  correctAnswer: "B"
},
{
  id: 464,
  question: "How do you force LLM to say “I don’t know”?",
  options: {
    A: "High temperature",
    B: "Explicit system instruction + empty-context handling",
    C: "Larger model",
    D: "Longer prompts"
  },
  correctAnswer: "B"
},
{
  id: 465,
  question: "Why is hallucination dangerous in production?",
  options: {
    A: "Looks unprofessional",
    B: "Confident wrong outputs mislead users and decisions",
    C: "Uses more tokens",
    D: "Slower responses"
  },
  correctAnswer: "B"
},
{
  id: 466,
  question: "RAG system returns irrelevant answers. FIRST thing to inspect?",
  options: {
    A: "LLM model",
    B: "Retrieved chunks from vector DB",
    C: "UI",
    D: "Temperature"
  },
  correctAnswer: "B"
},
{
  id: 467,
  question: "RAG retrieves correct docs but answer is still wrong. Likely cause?",
  options: {
    A: "Embedding model",
    B: "Prompt does not constrain model to use retrieved context",
    C: "Vector DB size",
    D: "Chunk overlap"
  },
  correctAnswer: "B"
},
{
  id: 468,
  question: "RAG answers outdated information. Root issue?",
  options: {
    A: "LLM memory",
    B: "Vector database not updated/re-indexed",
    C: "Chunk size too small",
    D: "Top-k too low"
  },
  correctAnswer: "B"
},
{
  id: 469,
  question: "Increasing top-k retrieval makes answers worse. Why?",
  options: {
    A: "Retrieval bug",
    B: "Too much irrelevant context added to prompt",
    C: "LLM limit reached",
    D: "Embedding mismatch"
  },
  correctAnswer: "B"
},
{
  id: 470,
  question: "RAG performs poorly on complex questions. Best improvement?",
  options: {
    A: "Larger chunks",
    B: "Multi-query or query-rewriting retrieval",
    C: "Increase temperature",
    D: "Remove overlap"
  },
  correctAnswer: "B"
},
{
  id: 471,
  question: "Model latency spikes under traffic. Best solution?",
  options: {
    A: "Retrain model",
    B: "Add caching, autoscaling, or smaller model",
    C: "Increase epochs",
    D: "Change metric"
  },
  correctAnswer: "B"
},
{
  id: 472,
  question: "Model predictions cannot be reproduced later. Missing piece?",
  options: {
    A: "GPU",
    B: "Model & data versioning",
    C: "Logging UI",
    D: "Batch size"
  },
  correctAnswer: "B"
},
{
  id: 473,
  question: "After retraining, model performance drops unexpectedly. Why?",
  options: {
    A: "New model worse",
    B: "Data leakage or data shift introduced",
    C: "Learning rate",
    D: "Tokenization"
  },
  correctAnswer: "B"
},
{
  id: 474,
  question: "You don’t have labels in production. How monitor model?",
  options: {
    A: "Training loss",
    B: "Input & prediction distribution drift monitoring",
    C: "Accuracy",
    D: "Precision"
  },
  correctAnswer: "B"
},
{
  id: 475,
  question: "Business demands instant rollback after bad deployment. Required system feature?",
  options: {
    A: "Large model",
    B: "Versioned deployments (blue-green / canary)",
    C: "High temperature",
    D: "More features"
  },
  correctAnswer: "B"
},
{
  id: 476,
  question: "Best signal of a strong AI engineer?",
  options: {
    A: "Knows algorithms",
    B: "Can diagnose failures across data, model, and system",
    C: "Trains big models",
    D: "Writes prompts only"
  },
  correctAnswer: "B"
},
{
  id: 477,
  question: "Model success in real world is measured by:",
  options: {
    A: "Validation accuracy",
    B: "Business KPIs & user impact",
    C: "Loss value",
    D: "GPU usage"
  },
  correctAnswer: "B"
},
{
  id: 478,
  question: "When users complain but metrics look good, what do you do?",
  options: {
    A: "Ignore users",
    B: "Investigate metric–reality gap & redefine success criteria",
    C: "Increase model size",
    D: "Change optimizer"
  },
  correctAnswer: "B"
},
{
  id: 479,
  question: "You must choose between slightly worse accuracy but 10× faster inference. Best choice?",
  options: {
    A: "Higher accuracy always",
    B: "Depends on product latency requirements & business impact",
    C: "Always faster",
    D: "Always accurate"
  },
  correctAnswer: "B"
},
{
  id: 480,
  question: "Final decision-maker in AI system design should be:",
  options: {
    A: "Model",
    B: "Engineer",
    C: "Business + engineering together",
    D: "LLM"
  },
  correctAnswer: "C"
},
{
  id: 481,
  question: "Why do we save trained ML models?",
  options: {
    A: "To improve accuracy",
    B: "To reuse them for inference without retraining",
    C: "To reduce dataset size",
    D: "To debug code"
  },
  correctAnswer: "B"
},
{
  id: 482,
  question: "What should be saved along with the model weights?",
  options: {
    A: "UI code",
    B: "Preprocessing logic & feature schema",
    C: "Training logs only",
    D: "GPU config"
  },
  correctAnswer: "B"
},
{
  id: 483,
  question: "A common mistake during model loading is:",
  options: {
    A: "Using pickle",
    B: "Forgetting to load preprocessing pipeline",
    C: "Using cloud storage",
    D: "Versioning models"
  },
  correctAnswer: "B"
},
{
  id: 484,
  question: "Model serialization formats include:",
  options: {
    A: "CSV",
    B: "Pickle / Joblib / ONNX / SavedModel",
    C: "JSON only",
    D: "TXT"
  },
  correctAnswer: "B"
},
{
  id: 485,
  question: "Why ONNX is useful?",
  options: {
    A: "Easier training",
    B: "Framework-agnostic model deployment",
    C: "Smaller datasets",
    D: "Better accuracy"
  },
  correctAnswer: "B"
},
{
  id: 486,
  question: "Why expose ML models via REST APIs?",
  options: {
    A: "For training",
    B: "To serve predictions to other systems",
    C: "For data labeling",
    D: "For monitoring only"
  },
  correctAnswer: "B"
},
{
  id: 487,
  question: "FastAPI is preferred over Flask for ML serving because:",
  options: {
    A: "It’s older",
    B: "Faster, async support, auto OpenAPI docs",
    C: "More plugins",
    D: "Less memory usage always"
  },
  correctAnswer: "B"
},
{
  id: 488,
  question: "A typical prediction API endpoint should:",
  options: {
    A: "Accept raw user text only",
    B: "Validate input, preprocess, predict, return output",
    C: "Train model",
    D: "Store embeddings"
  },
  correctAnswer: "B"
},
{
  id: 489,
  question: "Why input validation is critical?",
  options: {
    A: "Improves accuracy",
    B: "Prevents crashes & bad predictions",
    C: "Improves speed",
    D: "Improves UI"
  },
  correctAnswer: "B"
},
{
  id: 490,
  question: "Model inference APIs should be:",
  options: {
    A: "Stateful",
    B: "Stateless for scalability",
    C: "GPU-bound always",
    D: "Interactive"
  },
  correctAnswer: "B"
},
{
  id: 491,
  question: "Why Docker is used for ML deployment?",
  options: {
    A: "Faster training",
    B: "Consistent runtime environment across systems",
    C: "Better accuracy",
    D: "Smaller models"
  },
  correctAnswer: "B"
},
{
  id: 492,
  question: "Docker helps solve:",
  options: {
    A: "Data drift",
    B: "“Works on my machine” problem",
    C: "Overfitting",
    D: "Tokenization"
  },
  correctAnswer: "B"
},
{
  id: 493,
  question: "A Docker image contains:",
  options: {
    A: "Only code",
    B: "OS, dependencies, code, and runtime environment",
    C: "Model weights only",
    D: "Data"
  },
  correctAnswer: "B"
},
{
  id: 494,
  question: "Containers are preferred because they are:",
  options: {
    A: "Heavy",
    B: "Lightweight and reproducible",
    C: "Slower",
    D: "GPU-only"
  },
  correctAnswer: "B"
},
{
  id: 495,
  question: "In ML systems, Docker is mostly used for:",
  options: {
    A: "Data labeling",
    B: "Model serving & pipelines",
    C: "Feature engineering only",
    D: "Visualization"
  },
  correctAnswer: "B"
},
{
  id: 496,
  question: "CI/CD in ML means:",
  options: {
    A: "Continuous data labeling",
    B: "Continuous integration & deployment of code and models",
    C: "Continuous inference only",
    D: "Continuous training only"
  },
  correctAnswer: "B"
},
{
  id: 497,
  question: "CI in ML typically runs:",
  options: {
    A: "Model inference",
    B: "Tests for code, data checks, model validation",
    C: "Full retraining always",
    D: "GPU benchmarks"
  },
  correctAnswer: "B"
},
{
  id: 498,
  question: "CD ensures:",
  options: {
    A: "Code quality",
    B: "Automated, safe deployment of models to production",
    C: "Better accuracy",
    D: "Faster labeling"
  },
  correctAnswer: "B"
},
{
  id: 499,
  question: "A key difference between ML CI/CD and software CI/CD:",
  options: {
    A: "No tests in ML",
    B: "Models + data need versioning & validation",
    C: "No rollback in ML",
    D: "No pipelines"
  },
  correctAnswer: "B"
},
{
  id: 500,
  question: "A failed ML deployment should:",
  options: {
    A: "Stay live",
    B: "Automatically rollback to previous stable model",
    C: "Be ignored",
    D: "Increase traffic"
  },
  correctAnswer: "B"
},
{
  id: 501,
  question: "What should be logged in production ML systems?",
  options: {
    A: "Only errors",
    B: "Inputs, outputs, model version, timestamps",
    C: "Only accuracy",
    D: "GPU usage only"
  },
  correctAnswer: "B"
},
{
  id: 502,
  question: "Logging predictions helps to:",
  options: {
    A: "Improve UI",
    B: "Debug issues & support retraining",
    C: "Reduce latency",
    D: "Improve embeddings"
  },
  correctAnswer: "B"
},
{
  id: 503,
  question: "Which metric is hard to monitor in production?",
  options: {
    A: "Latency",
    B: "Throughput",
    C: "Accuracy (due to delayed labels)",
    D: "Error rate"
  },
  correctAnswer: "C"
},
{
  id: 504,
  question: "Why monitor input data distribution?",
  options: {
    A: "Improve UI",
    B: "Detect data drift early",
    C: "Reduce cost",
    D: "Increase accuracy automatically"
  },
  correctAnswer: "B"
},
{
  id: 505,
  question: "Monitoring latency is important because:",
  options: {
    A: "Affects model training",
    B: "Directly impacts user experience & SLAs",
    C: "Improves embeddings",
    D: "Reduces drift"
  },
  correctAnswer: "B"
},
{
  id: 506,
  question: "If model service crashes, best practice is:",
  options: {
    A: "Restart manually",
    B: "Health checks + auto-restart + fallback",
    C: "Retrain model",
    D: "Ignore"
  },
  correctAnswer: "B"
},
{
  id: 507,
  question: "Fallback strategy means:",
  options: {
    A: "Always use latest model",
    B: "Use simpler rule-based or previous model if AI fails",
    C: "Shut down system",
    D: "Ask user to wait"
  },
  correctAnswer: "B"
},
{
  id: 508,
  question: "Graceful degradation ensures:",
  options: {
    A: "System stops completely",
    B: "Partial functionality instead of total failure",
    C: "Faster training",
    D: "Better accuracy"
  },
  correctAnswer: "B"
},
{
  id: 509,
  question: "Rate limiting is used to:",
  options: {
    A: "Improve accuracy",
    B: "Protect ML APIs from abuse & overload",
    C: "Improve embeddings",
    D: "Reduce drift"
  },
  correctAnswer: "B"
},
{
  id: 510,
  question: "Circuit breakers help by:",
  options: {
    A: "Training faster",
    B: "Preventing cascading failures when service is unhealthy",
    C: "Increasing throughput",
    D: "Improving metrics"
  },
  correctAnswer: "B"
},
{
  id: 511,
  question: "Model API works locally but fails in cloud. Likely issue:",
  options: {
    A: "Model size",
    B: "Missing dependencies / environment mismatch",
    C: "Learning rate",
    D: "Tokenization"
  },
  correctAnswer: "B"
},
{
  id: 512,
  question: "Predictions suddenly become NaN. Likely cause:",
  options: {
    A: "Model too large",
    B: "Bad input values not validated",
    C: "High temperature",
    D: "Docker bug"
  },
  correctAnswer: "B"
},
{
  id: 513,
  question: "After new deployment, error rate spikes. What to do FIRST?",
  options: {
    A: "Retrain model",
    B: "Roll back to previous stable version",
    C: "Increase replicas",
    D: "Change optimizer"
  },
  correctAnswer: "B"
},
{
  id: 514,
  question: "Production ML system must prioritize:",
  options: {
    A: "Latest model",
    B: "Reliability & user trust",
    C: "Research metrics",
    D: "Training speed"
  },
  correctAnswer: "B"
},
{
  id: 515,
  question: "You need to support multiple model versions simultaneously. Use:",
  options: {
    A: "Hard overwrite",
    B: "Versioned endpoints / routing",
    C: "Single endpoint",
    D: "Batch inference"
  },
  correctAnswer: "B"
},
{
  id: 516,
  question: "A low-maturity ML system:",
  options: {
    A: "Has Docker",
    B: "Manual deployment, no monitoring",
    C: "Versioned models",
    D: "Automated rollback"
  },
  correctAnswer: "B"
},
{
  id: 517,
  question: "A mature ML system includes:",
  options: {
    A: "Big models",
    B: "Monitoring, rollback, CI/CD, logging",
    C: "Only notebooks",
    D: "High temperature"
  },
  correctAnswer: "B"
},
{
  id: 518,
  question: "Which is NOT essential at start?",
  options: {
    A: "Logging",
    B: "Monitoring",
    C: "Full auto-retraining pipelines",
    D: "Versioning"
  },
  correctAnswer: "C"
},
{
  id: 519,
  question: "Why automation is important in ML deployment?",
  options: {
    A: "Reduce coding",
    B: "Reduce human error & speed iteration",
    C: "Improve accuracy",
    D: "Improve embeddings"
  },
  correctAnswer: "B"
},
{
  id: 520,
  question: "First production ML priority should be:",
  options: {
    A: "Perfect accuracy",
    B: "Stability & observability",
    C: "Latest framework",
    D: "GPU usage"
  },
  correctAnswer: "B"
},
{
  id: 521,
  question: "ML deployment fails mostly because of:",
  options: {
    A: "Algorithms",
    B: "Poor engineering practices & missing safeguards",
    C: "Low data volume",
    D: "Optimizers"
  },
  correctAnswer: "B"
},
{
  id: 522,
  question: "When model output impacts users directly, you must ensure:",
  options: {
    A: "High temperature",
    B: "Monitoring, fallback, and rollback mechanisms",
    C: "Fast training",
    D: "Fine-tuning"
  },
  correctAnswer: "B"
},
{
  id: 523,
  question: "Which tool helps catch issues BEFORE production?",
  options: {
    A: "Monitoring",
    B: "CI tests & staging environments",
    C: "Logs",
    D: "Rollback"
  },
  correctAnswer: "B"
},
{
  id: 524,
  question: "Why separate training and serving containers?",
  options: {
    A: "Easier coding",
    B: "Different dependencies & resource needs",
    C: "Lower cost",
    D: "Better accuracy"
  },
  correctAnswer: "B"
},
{
  id: 525,
  question: "Logging without monitoring is:",
  options: {
    A: "Enough",
    B: "Not actionable in real-time",
    C: "Cheaper",
    D: "Faster"
  },
  correctAnswer: "B"
},
{
  id: 526,
  question: "Production ML is successful when:",
  options: {
    A: "Model is accurate",
    B: "System is reliable, observable, and recoverable",
    C: "Code is short",
    D: "GPU is powerful"
  },
  correctAnswer: "B"
},
{
  id: 527,
  question: "AI engineers must understand MLOps because:",
  options: {
    A: "It’s DevOps",
    B: "Models are useless if they can’t run reliably in production",
    C: "It’s trendy",
    D: "It’s optional"
  },
  correctAnswer: "B"
},
{
  id: 528,
  question: "Best deployment mindset:",
  options: {
    A: "Deploy fast, fix later",
    B: "Deploy safely, monitor, iterate",
    C: "Never deploy",
    D: "Train bigger models"
  },
  correctAnswer: "B"
},
{
  id: 529,
  question: "Final ownership of production AI system lies with:",
  options: {
    A: "LLM",
    B: "Platform team only",
    C: "AI engineers + platform together",
    D: "Users"
  },
  correctAnswer: "C"
},
{
  id: 530,
  question: "A strong mid-level AI engineer can:",
  options: {
    A: "Train models",
    B: "Ship, monitor, debug, and recover AI systems in production",
    C: "Write prompts only",
    D: "Tune hyperparameters"
  },
  correctAnswer: "B"
},
,
  {
    id: 531,
    question: "What is Artificial Intelligence?",
    options: {
      A: "A database system",
      B: "A machine that mimics human intelligence",
      C: "A programming language",
      D: "A hardware component",
    },
    correctAnswer: "B",
  },
  {
    id: 532,
    question: "Which of the following is a subset of AI?",
    options: {
      A: "Machine Learning",
      B: "Cloud Computing",
      C: "Cyber Security",
      D: "Networking",
    },
    correctAnswer: "A",
  },
  {
    id: 533,
    question: "What is Machine Learning?",
    options: {
      A: "Explicitly programming rules",
      B: "Machines learning from data",
      C: "Storing data in memory",
      D: "Building hardware",
    },
    correctAnswer: "B",
  },
  {
    id: 534,
    question: "Which learning type uses labeled data?",
    options: {
      A: "Unsupervised Learning",
      B: "Reinforcement Learning",
      C: "Supervised Learning",
      D: "Deep Learning",
    },
    correctAnswer: "C",
  },
  {
    id: 535,
    question: "Which learning type finds hidden patterns?",
    options: {
      A: "Supervised Learning",
      B: "Unsupervised Learning",
      C: "Reinforcement Learning",
      D: "Transfer Learning",
    },
    correctAnswer: "B",
  },
  {
    id: 536,
    question: "Which learning uses reward and punishment?",
    options: {
      A: "Supervised Learning",
      B: "Unsupervised Learning",
      C: "Reinforcement Learning",
      D: "Deep Learning",
    },
    correctAnswer: "C",
  },
  {
    id: 537,
    question: "What is Deep Learning?",
    options: {
      A: "Learning from big books",
      B: "Machine learning using deep neural networks",
      C: "Learning from rules",
      D: "Data mining",
    },
    correctAnswer: "B",
  },
  {
    id: 538,
    question: "Which neural network is used for image processing?",
    options: {
      A: "RNN",
      B: "CNN",
      C: "LSTM",
      D: "GAN",
    },
    correctAnswer: "B",
  },
  {
    id: 539,
    question: "Which neural network is used for sequence data?",
    options: {
      A: "CNN",
      B: "RNN",
      C: "Autoencoder",
      D: "GAN",
    },
    correctAnswer: "B",
  },
  {
    id: 540,
    question: "What is an epoch?",
    options: {
      A: "One data sample",
      B: "One complete pass of dataset",
      C: "One neural layer",
      D: "One parameter update",
    },
    correctAnswer: "B",
  },
  {
    id: 541,
    question: "What is a batch?",
    options: {
      A: "Complete dataset",
      B: "Single data point",
      C: "Subset of dataset",
      D: "Model checkpoint",
    },
    correctAnswer: "C",
  },
  {
    id: 542,
    question: "Which activation function is commonly used?",
    options: {
      A: "Sigmoid",
      B: "ReLU",
      C: "Softmax",
      D: "Tanh",
    },
    correctAnswer: "B",
  },
  {
    id: 543,
    question: "What does LLM stand for?",
    options: {
      A: "Low Level Model",
      B: "Large Language Model",
      C: "Logical Learning Machine",
      D: "Linear Language Model",
    },
    correctAnswer: "B",
  },
  {
    id: 544,
    question: "Which is an example of LLM?",
    options: {
      A: "ResNet",
      B: "YOLO",
      C: "GPT",
      D: "VGG",
    },
    correctAnswer: "C",
  },
  {
    id: 545,
    question: "Which architecture is used in most LLMs?",
    options: {
      A: "CNN",
      B: "RNN",
      C: "Transformer",
      D: "Autoencoder",
    },
    correctAnswer: "C",
  },
  {
    id: 546,
    question: "What is Prompt Engineering?",
    options: {
      A: "Designing hardware prompts",
      B: "Designing effective inputs for LLMs",
      C: "Optimizing databases",
      D: "Training neural networks",
    },
    correctAnswer: "B",
  },
  {
    id: 547,
    question: "Which prompt gives better output?",
    options: {
      A: "Explain AI",
      B: "AI",
      C: "Explain AI with examples in simple language",
      D: "Tell something",
    },
    correctAnswer: "C",
  },
  {
    id: 548,
    question: "What is RAG?",
    options: {
      A: "Random AI Generator",
      B: "Retrieval Augmented Generation",
      C: "Recursive AI Generator",
      D: "Real-time AI Gateway",
    },
    correctAnswer: "B",
  },
  {
    id: 549,
    question: "Why is RAG used?",
    options: {
      A: "To train models faster",
      B: "To reduce model size",
      C: "To combine external knowledge with LLMs",
      D: "To remove hallucinations completely",
    },
    correctAnswer: "C",
  },
  {
    id: 550,
    question: "What is a Vector Database?",
    options: {
      A: "Relational database",
      B: "Database for images",
      C: "Database for embeddings and similarity search",
      D: "Database for logs",
    },
    correctAnswer: "C",
  },
  {
    id: 551,
    question: "What are embeddings?",
    options: {
      A: "Images",
      B: "Audio signals",
      C: "Numerical representation of data",
      D: "Database indexes",
    },
    correctAnswer: "C",
  },
  {
    id: 552,
    question: "Which is a vector database?",
    options: {
      A: "MySQL",
      B: "MongoDB",
      C: "FAISS",
      D: "Redis",
    },
    correctAnswer: "C",
  },
  {
    id: 553,
    question: "What is Generative AI?",
    options: {
      A: "AI that predicts labels",
      B: "AI that classifies data",
      C: "AI that generates new content",
      D: "AI that compresses files",
    },
    correctAnswer: "C",
  },
  {
    id: 554,
    question: "Which is a Generative AI model?",
    options: {
      A: "BERT",
      B: "GAN",
      C: "XGBoost",
      D: "SVM",
    },
    correctAnswer: "B",
  },
  {
    id: 555,
    question: "Which task is Generative AI used for?",
    options: {
      A: "Fraud detection",
      B: "Text generation",
      C: "Classification",
      D: "Clustering",
    },
    correctAnswer: "B",
  },
  {
    id: 556,
    question: "What is fine-tuning?",
    options: {
      A: "Training from scratch",
      B: "Improving a pre-trained model",
      C: "Removing layers",
      D: "Compressing model",
    },
    correctAnswer: "B",
  },
  {
    id: 557,
    question: "What is overfitting?",
    options: {
      A: "Model performs well on new data",
      B: "Model memorizes training data",
      C: "Model trains fast",
      D: "Model is simple",
    },
    correctAnswer: "B",
  },
  {
    id: 558,
    question: "Which loss function is used for classification?",
    options: {
      A: "MSE",
      B: "MAE",
      C: "Cross Entropy",
      D: "RMSE",
    },
    correctAnswer: "C",
  },
  {
    id: 559,
    question: "Which optimizer is commonly used?",
    options: {
      A: "Naive Bayes",
      B: "Adam",
      C: "KNN",
      D: "Decision Tree",
    },
    correctAnswer: "B",
  },
  {
    id: 560,
    question: "What is transfer learning?",
    options: {
      A: "Copying data",
      B: "Training from scratch",
      C: "Using pre-trained model for new task",
      D: "Deleting layers",
    },
    correctAnswer: "C",
  },
  {
    id: 561,
    question: "Which algorithm is used for regression?",
    options: {
      A: "KNN",
      B: "Linear Regression",
      C: "K-Means",
      D: "Apriori",
    },
    correctAnswer: "B",
  },
  {
    id: 562,
    question: "Which algorithm is used for clustering?",
    options: {
      A: "Logistic Regression",
      B: "Decision Tree",
      C: "K-Means",
      D: "SVM",
    },
    correctAnswer: "C",
  },
  {
    id: 563,
    question: "Which model is used for binary classification?",
    options: {
      A: "Linear Regression",
      B: "Logistic Regression",
      C: "K-Means",
      D: "PCA",
    },
    correctAnswer: "B",
  },
  {
    id: 564,
    question: "What is PCA used for?",
    options: {
      A: "Classification",
      B: "Regression",
      C: "Dimensionality Reduction",
      D: "Clustering",
    },
    correctAnswer: "C",
  },
  {
    id: 565,
    question: "Which algorithm is distance-based?",
    options: {
      A: "Decision Tree",
      B: "Naive Bayes",
      C: "KNN",
      D: "Linear Regression",
    },
    correctAnswer: "C",
  },
  {
    id: 566,
    question: "Which algorithm is probabilistic?",
    options: {
      A: "SVM",
      B: "KNN",
      C: "Naive Bayes",
      D: "Decision Tree",
    },
    correctAnswer: "C",
  },
  {
    id: 567,
    question: "What is a confusion matrix?",
    options: {
      A: "Training dataset",
      B: "Performance evaluation table",
      C: "Loss function",
      D: "Optimizer",
    },
    correctAnswer: "B",
  },
  {
    id: 568,
    question: "Which metric measures model precision?",
    options: {
      A: "Recall",
      B: "Accuracy",
      C: "Precision",
      D: "F1-score",
    },
    correctAnswer: "C",
  },
  {
    id: 569,
    question: "Which metric balances precision and recall?",
    options: {
      A: "Accuracy",
      B: "Recall",
      C: "Precision",
      D: "F1-score",
    },
    correctAnswer: "D",
  },
  {
    id: 570,
    question: "Which metric is used for regression?",
    options: {
      A: "Accuracy",
      B: "F1-score",
      C: "RMSE",
      D: "Recall",
    },
    correctAnswer: "C",
  },
  {
    id: 571,
    question: "What is normalization?",
    options: {
      A: "Removing missing values",
      B: "Scaling data to a range",
      C: "Encoding labels",
      D: "Splitting dataset",
    },
    correctAnswer: "B",
  },
  {
    id: 572,
    question: "What is feature engineering?",
    options: {
      A: "Model tuning",
      B: "Data collection",
      C: "Creating and selecting useful features",
      D: "Model deployment",
    },
    correctAnswer: "C",
  },
  {
    id: 573,
    question: "What is train-test split?",
    options: {
      A: "Splitting model layers",
      B: "Splitting dataset for training and testing",
      C: "Splitting features",
      D: "Splitting labels",
    },
    correctAnswer: "B",
  },
  {
    id: 574,
    question: "What is cross-validation?",
    options: {
      A: "Model deployment",
      B: "Repeated model evaluation on different data splits",
      C: "Feature selection",
      D: "Data cleaning",
    },
    correctAnswer: "B",
  },
  {
    id: 575,
    question: "What is hyperparameter tuning?",
    options: {
      A: "Changing training data",
      B: "Optimizing model parameters",
      C: "Selecting best model settings",
      D: "Reducing dataset size",
    },
    correctAnswer: "C",
  },
  {
    id: 576,
    question: "Which technique reduces overfitting?",
    options: {
      A: "More epochs",
      B: "Dropout",
      C: "Higher learning rate",
      D: "More layers",
    },
    correctAnswer: "B",
  },
  {
    id: 577,
    question: "What is early stopping?",
    options: {
      A: "Stopping data collection",
      B: "Stopping training when performance degrades",
      C: "Stopping model deployment",
      D: "Stopping evaluation",
    },
    correctAnswer: "B",
  },
  {
    id: 578,
    question: "What is gradient descent?",
    options: {
      A: "Loss function",
      B: "Optimization algorithm",
      C: "Activation function",
      D: "Evaluation metric",
    },
    correctAnswer: "B",
  },
  {
    id: 579,
    question: "What is learning rate?",
    options: {
      A: "Speed of data loading",
      B: "Speed of model inference",
      C: "Step size during optimization",
      D: "Batch size",
    },
    correctAnswer: "C",
  },
  {
    id: 580,
    question: "Which layer produces final output?",
    options: {
      A: "Input layer",
      B: "Hidden layer",
      C: "Output layer",
      D: "Dropout layer",
    },
    correctAnswer: "C",
  },
  {
    id: 581,
    question: "What is backpropagation?",
    options: {
      A: "Forward pass",
      B: "Updating weights using error",
      C: "Data preprocessing",
      D: "Feature extraction",
    },
    correctAnswer: "B",
  },
  {
    id: 582,
    question: "Which layer extracts features in CNN?",
    options: {
      A: "Dense layer",
      B: "Pooling layer",
      C: "Convolution layer",
      D: "Dropout layer",
    },
    correctAnswer: "C",
  },
  {
    id: 583,
    question: "What is pooling?",
    options: {
      A: "Feature extraction",
      B: "Dimensionality reduction",
      C: "Classification",
      D: "Normalization",
    },
    correctAnswer: "B",
  },
  {
    id: 584,
    question: "Which pooling reduces size by taking max value?",
    options: {
      A: "Average Pooling",
      B: "Max Pooling",
      C: "Global Pooling",
      D: "Sum Pooling",
    },
    correctAnswer: "B",
  },
  {
    id: 585,
    question: "What is an autoencoder?",
    options: {
      A: "Classification model",
      B: "Regression model",
      C: "Unsupervised feature learning model",
      D: "Reinforcement model",
    },
    correctAnswer: "C",
  },
  {
    id: 586,
    question: "What is an LSTM?",
    options: {
      A: "Image model",
      B: "Sequence model",
      C: "Clustering model",
      D: "Regression model",
    },
    correctAnswer: "B",
  },
  {
    id: 587,
    question: "Which model is used for text generation?",
    options: {
      A: "Random Forest",
      B: "Transformer",
      C: "KNN",
      D: "SVM",
    },
    correctAnswer: "B",
  },
  {
    id: 588,
    question: "What is tokenization?",
    options: {
      A: "Encrypting data",
      B: "Splitting text into units",
      C: "Removing stopwords",
      D: "Encoding labels",
    },
    correctAnswer: "B",
  },
  {
    id: 589,
    question: "What is a token?",
    options: {
      A: "A word or subword unit",
      B: "A database record",
      C: "A neural layer",
      D: "A loss value",
    },
    correctAnswer: "A",
  },
  {
    id: 590,
    question: "What is temperature in LLMs?",
    options: {
      A: "Training speed",
      B: "Model size",
      C: "Controls randomness of output",
      D: "Learning rate",
    },
    correctAnswer: "C",
  },
  {
    id: 591,
    question: "What is hallucination in LLMs?",
    options: {
      A: "Correct output",
      B: "Model crash",
      C: "Incorrect or fabricated response",
      D: "Training error",
    },
    correctAnswer: "C",
  },
  {
    id: 592,
    question: "How can hallucination be reduced?",
    options: {
      A: "Using bigger model",
      B: "Using RAG",
      C: "Increasing temperature",
      D: "Reducing epochs",
    },
    correctAnswer: "B",
  },
  {
    id: 593,
    question: "What is a prompt?",
    options: {
      A: "Training data",
      B: "User input to LLM",
      C: "Model output",
      D: "Loss value",
    },
    correctAnswer: "B",
  },
  {
    id: 594,
    question: "What is zero-shot prompting?",
    options: {
      A: "No model training",
      B: "Prompt without examples",
      C: "Prompt with many examples",
      D: "Prompt with code",
    },
    correctAnswer: "B",
  },
  {
    id: 595,
    question: "What is few-shot prompting?",
    options: {
      A: "Prompt without examples",
      B: "Prompt with many examples",
      C: "Prompt with few examples",
      D: "Prompt with code",
    },
    correctAnswer: "C",
  },
  {
    id: 596,
    question: "What is chain-of-thought prompting?",
    options: {
      A: "Prompting with reasoning steps",
      B: "Prompting with images",
      C: "Prompting with code",
      D: "Prompting with tables",
    },
    correctAnswer: "A",
  },
  {
    id: 597,
    question: "What is context window?",
    options: {
      A: "Training dataset size",
      B: "Maximum input size of LLM",
      C: "Output length",
      D: "Model depth",
    },
    correctAnswer: "B",
  },
  {
    id: 598,
    question: "What is chunking in RAG?",
    options: {
      A: "Splitting documents into smaller parts",
      B: "Encrypting data",
      C: "Compressing embeddings",
      D: "Removing stopwords",
    },
    correctAnswer: "A",
  },
  {
    id: 599,
    question: "What is similarity search?",
    options: {
      A: "Exact match search",
      B: "Searching similar embeddings",
      C: "Keyword search",
      D: "Regex search",
    },
    correctAnswer: "B",
  },
  {
    id: 600,
    question: "Which distance metric is commonly used?",
    options: {
      A: "Manhattan",
      B: "Euclidean",
      C: "Cosine similarity",
      D: "Hamming",
    },
    correctAnswer: "C",
  },
  {
    id: 601,
    question: "Which library is used for embeddings?",
    options: {
      A: "TensorFlow",
      B: "OpenAI",
      C: "Scikit-learn",
      D: "Pandas",
    },
    correctAnswer: "B",
  },
  {
    id: 602,
    question: "Which database supports vector search?",
    options: {
      A: "PostgreSQL",
      B: "FAISS",
      C: "SQLite",
      D: "Oracle",
    },
    correctAnswer: "B",
  },
  {
    id: 603,
    question: "What is indexing in vector DB?",
    options: {
      A: "Sorting data",
      B: "Creating search structure",
      C: "Encrypting embeddings",
      D: "Compressing vectors",
    },
    correctAnswer: "B",
  },
  {
    id: 604,
    question: "Which model generates images?",
    options: {
      A: "BERT",
      B: "GAN",
      C: "KNN",
      D: "SVM",
    },
    correctAnswer: "B",
  },
  {
    id: 605,
    question: "What is diffusion model?",
    options: {
      A: "Classification model",
      B: "Regression model",
      C: "Image generation model",
      D: "Clustering model",
    },
    correctAnswer: "C",
  },
  {
    id: 606,
    question: "Which AI can generate music?",
    options: {
      A: "Discriminative AI",
      B: "Predictive AI",
      C: "Generative AI",
      D: "Rule-based AI",
    },
    correctAnswer: "C",
  },
  {
    id: 607,
    question: "What is model inference?",
    options: {
      A: "Training model",
      B: "Evaluating model",
      C: "Using model for prediction",
      D: "Deploying model",
    },
    correctAnswer: "C",
  },
  {
    id: 608,
    question: "What is model deployment?",
    options: {
      A: "Training model",
      B: "Testing model",
      C: "Making model available for users",
      D: "Saving model",
    },
    correctAnswer: "C",
  },
  {
    id: 609,
    question: "What is MLOps?",
    options: {
      A: "Model training",
      B: "Model deployment and monitoring",
      C: "Model architecture",
      D: "Model optimization",
    },
    correctAnswer: "B",
  },
  {
    id: 610,
    question: "What is data drift?",
    options: {
      A: "Model bug",
      B: "Change in data distribution",
      C: "Training error",
      D: "Deployment failure",
    },
    correctAnswer: "B",
  },
  {
    id: 611,
    question: "What is concept drift?",
    options: {
      A: "Model architecture change",
      B: "Change in target relationship",
      C: "Change in optimizer",
      D: "Change in learning rate",
    },
    correctAnswer: "B",
  },
  {
    id: 612,
    question: "What is monitoring in MLOps?",
    options: {
      A: "Training model",
      B: "Tracking model performance",
      C: "Feature engineering",
      D: "Data cleaning",
    },
    correctAnswer: "B",
  },
  {
    id: 613,
    question: "What is A/B testing?",
    options: {
      A: "Comparing two models",
      B: "Training two models",
      C: "Testing two datasets",
      D: "Splitting features",
    },
    correctAnswer: "A",
  },
  {
    id: 614,
    question: "What is shadow deployment?",
    options: {
      A: "Hidden model",
      B: "Testing new model alongside old one",
      C: "Deleting model",
      D: "Training secretly",
    },
    correctAnswer: "B",
  },
  {
    id: 615,
    question: "What is explainable AI (XAI)?",
    options: {
      A: "AI that explains its predictions",
      B: "AI that trains faster",
      C: "AI that runs on CPU",
      D: "AI that generates text",
    },
    correctAnswer: "A",
  },
  {
    id: 616,
    question: "Which tool is used for model explainability?",
    options: {
      A: "SHAP",
      B: "TensorFlow",
      C: "PyTorch",
      D: "Keras",
    },
    correctAnswer: "A",
  },
  {
    id: 617,
    question: "What is bias in AI?",
    options: {
      A: "Model speed",
      B: "Unfair predictions",
      C: "Training error",
      D: "Loss value",
    },
    correctAnswer: "B",
  },
  {
    id: 618,
    question: "What is fairness in AI?",
    options: {
      A: "Fast inference",
      B: "Equal predictions",
      C: "Ethical predictions",
      D: "Accurate predictions",
    },
    correctAnswer: "C",
  },
  {
    id: 619,
    question: "What is data labeling?",
    options: {
      A: "Data cleaning",
      B: "Adding target values",
      C: "Feature scaling",
      D: "Data encryption",
    },
    correctAnswer: "B",
  },
  {
    id: 620,
    question: "What is annotation?",
    options: {
      A: "Model training",
      B: "Data labeling",
      C: "Feature extraction",
      D: "Model deployment",
    },
    correctAnswer: "B",
  },
  {
    id: 621,
    question: "Which format is used for dataset storage?",
    options: {
      A: "CSV",
      B: "HTML",
      C: "MP3",
      D: "PNG",
    },
    correctAnswer: "A",
  },
  {
    id: 622,
    question: "What is ETL?",
    options: {
      A: "Model training",
      B: "Extract Transform Load",
      C: "Evaluation Technique",
      D: "Encoding Training Logic",
    },
    correctAnswer: "B",
  },
  {
    id: 623,
    question: "What is data pipeline?",
    options: {
      A: "Model architecture",
      B: "Flow of data processing",
      C: "Training loop",
      D: "Loss function",
    },
    correctAnswer: "B",
  },
  {
    id: 624,
    question: "What is streaming data?",
    options: {
      A: "Stored data",
      B: "Batch data",
      C: "Real-time data",
      D: "Historical data",
    },
    correctAnswer: "C",
  },
  {
    id: 625,
    question: "What is batch processing?",
    options: {
      A: "Real-time processing",
      B: "Processing data in groups",
      C: "Single record processing",
      D: "Streaming",
    },
    correctAnswer: "B",
  },
  {
    id: 626,
    question: "What is edge AI?",
    options: {
      A: "AI in cloud",
      B: "AI on local devices",
      C: "AI in data center",
      D: "AI on GPU",
    },
    correctAnswer: "B",
  },
  {
    id: 627,
    question: "Which device uses edge AI?",
    options: {
      A: "Cloud server",
      B: "Smartphone",
      C: "Database",
      D: "Web browser",
    },
    correctAnswer: "B",
  },
  {
    id: 628,
    question: "What is federated learning?",
    options: {
      A: "Centralized training",
      B: "Distributed training without sharing data",
      C: "Model compression",
      D: "Transfer learning",
    },
    correctAnswer: "B",
  },
  {
    id: 629,
    question: "What is model compression?",
    options: {
      A: "Increasing model size",
      B: "Reducing model size",
      C: "Adding layers",
      D: "Training longer",
    },
    correctAnswer: "B",
  },
  {
    id: 630,
    question: "What is quantization?",
    options: {
      A: "Increasing precision",
      B: "Reducing precision to save size",
      C: "Training model",
      D: "Deploying model",
    },
    correctAnswer: "B",
  },
  {
    id: 631,
    question: "What is pruning?",
    options: {
      A: "Removing unimportant weights",
      B: "Adding new layers",
      C: "Training faster",
      D: "Increasing accuracy",
    },
    correctAnswer: "A",
  },
  {
    id: 632,
    question: "What is knowledge distillation?",
    options: {
      A: "Large model teaching smaller model",
      B: "Small model teaching large model",
      C: "Model fine-tuning",
      D: "Model deployment",
    },
    correctAnswer: "A",
  },
  {
    id: 633,
    question: "What is multimodal AI?",
    options: {
      A: "Single data type AI",
      B: "AI using multiple data types",
      C: "Only text AI",
      D: "Only image AI",
    },
    correctAnswer: "B",
  },
  {
    id: 634,
    question: "Which is multimodal input?",
    options: {
      A: "Text only",
      B: "Image only",
      C: "Text and image",
      D: "Audio only",
    },
    correctAnswer: "C",
  },
  {
    id: 635,
    question: "What is speech recognition?",
    options: {
      A: "Text generation",
      B: "Image detection",
      C: "Converting speech to text",
      D: "Music generation",
    },
    correctAnswer: "C",
  },
  {
    id: 636,
    question: "What is text-to-speech?",
    options: {
      A: "Speech recognition",
      B: "Converting text to audio",
      C: "Translation",
      D: "Summarization",
    },
    correctAnswer: "B",
  },
  {
    id: 637,
    question: "What is NLP?",
    options: {
      A: "Natural Language Processing",
      B: "Neural Learning Process",
      C: "Network Logic Protocol",
      D: "Neural Language Programming",
    },
    correctAnswer: "A",
  },
  {
    id: 638,
    question: "Which task is NLP?",
    options: {
      A: "Image classification",
      B: "Speech recognition",
      C: "Text summarization",
      D: "Face detection",
    },
    correctAnswer: "C",
  },
  {
    id: 639,
    question: "What is sentiment analysis?",
    options: {
      A: "Detecting emotion in text",
      B: "Image recognition",
      C: "Speech generation",
      D: "Translation",
    },
    correctAnswer: "A",
  },
  {
    id: 640,
    question: "What is named entity recognition?",
    options: {
      A: "Finding keywords",
      B: "Finding names, places, organizations",
      C: "Text translation",
      D: "Speech recognition",
    },
    correctAnswer: "B",
  },
  {
    id: 641,
    question: "What is machine translation?",
    options: {
      A: "Speech recognition",
      B: "Text summarization",
      C: "Language translation",
      D: "Sentiment analysis",
    },
    correctAnswer: "C",
  },
  {
    id: 642,
    question: "What is chatbot?",
    options: {
      A: "Search engine",
      B: "AI conversational agent",
      C: "Database",
      D: "Compiler",
    },
    correctAnswer: "B",
  },
  {
    id: 643,
    question: "What is intent detection?",
    options: {
      A: "Image classification",
      B: "Understanding user purpose",
      C: "Speech synthesis",
      D: "Translation",
    },
    correctAnswer: "B",
  },
  {
    id: 644,
    question: "What is dialogue management?",
    options: {
      A: "Speech recognition",
      B: "Handling conversation flow",
      C: "Text summarization",
      D: "Translation",
    },
    correctAnswer: "B",
  },
  {
    id: 645,
    question: "What is recommendation system?",
    options: {
      A: "Search engine",
      B: "AI suggesting items",
      C: "Image classifier",
      D: "Speech recognizer",
    },
    correctAnswer: "B",
  },
  {
    id: 646,
    question: "Which algorithm is used in recommendation?",
    options: {
      A: "Collaborative filtering",
      B: "Decision tree",
      C: "Naive Bayes",
      D: "Linear regression",
    },
    correctAnswer: "A",
  },
  {
    id: 647,
    question: "What is content-based filtering?",
    options: {
      A: "User similarity",
      B: "Item similarity",
      C: "Random recommendation",
      D: "Rule-based",
    },
    correctAnswer: "B",
  },
  {
    id: 648,
    question: "What is collaborative filtering?",
    options: {
      A: "Based on item features",
      B: "Based on user behavior",
      C: "Based on rules",
      D: "Based on text",
    },
    correctAnswer: "B",
  },
  {
    id: 649,
    question: "What is cold start problem?",
    options: {
      A: "Model training issue",
      B: "New user/item with no data",
      C: "Server failure",
      D: "Slow inference",
    },
    correctAnswer: "B",
  },
  {
    id: 650,
    question: "What is anomaly detection?",
    options: {
      A: "Finding normal data",
      B: "Finding unusual patterns",
      C: "Classification",
      D: "Clustering",
    },
    correctAnswer: "B",
  },
  {
    id: 651,
    question: "Which algorithm is used for anomaly detection?",
    options: {
      A: "Isolation Forest",
      B: "Linear Regression",
      C: "Naive Bayes",
      D: "SVM",
    },
    correctAnswer: "A",
  },
  {
    id: 652,
    question: "What is fraud detection?",
    options: {
      A: "Image recognition",
      B: "Detecting abnormal transactions",
      C: "Speech recognition",
      D: "Text summarization",
    },
    correctAnswer: "B",
  },
  {
    id: 653,
    question: "What is time series forecasting?",
    options: {
      A: "Image prediction",
      B: "Predicting future values over time",
      C: "Text generation",
      D: "Clustering",
    },
    correctAnswer: "B",
  },
  {
    id: 654,
    question: "Which model is used for time series?",
    options: {
      A: "ARIMA",
      B: "KNN",
      C: "SVM",
      D: "Naive Bayes",
    },
    correctAnswer: "A",
  },
  {
    id: 655,
    question: "What is forecasting?",
    options: {
      A: "Classification",
      B: "Regression",
      C: "Predicting future",
      D: "Clustering",
    },
    correctAnswer: "C",
  },
  {
    id: 656,
    question: "What is computer vision?",
    options: {
      A: "Text processing",
      B: "Image and video understanding",
      C: "Speech recognition",
      D: "Audio generation",
    },
    correctAnswer: "B",
  },
  {
    id: 657,
    question: "Which task is computer vision?",
    options: {
      A: "Text translation",
      B: "Face detection",
      C: "Sentiment analysis",
      D: "Speech synthesis",
    },
    correctAnswer: "B",
  },
  {
    id: 658,
    question: "What is object detection?",
    options: {
      A: "Image classification",
      B: "Finding objects and their locations",
      C: "Image segmentation",
      D: "Video streaming",
    },
    correctAnswer: "B",
  },
  {
    id: 659,
    question: "Which model is used for object detection?",
    options: {
      A: "ResNet",
      B: "YOLO",
      C: "BERT",
      D: "GPT",
    },
    correctAnswer: "B",
  },
  {
    id: 660,
    question: "What is image segmentation?",
    options: {
      A: "Classifying image",
      B: "Dividing image into regions",
      C: "Resizing image",
      D: "Enhancing image",
    },
    correctAnswer: "B",
  },
  {
    id: 661,
    question: "What is supervised learning?",
    options: {
      A: "Learning without labels",
      B: "Learning with reward signals",
      C: "Learning with labeled data",
      D: "Learning with random data",
    },
    correctAnswer: "C",
  },
  {
    id: 662,
    question: "What is unsupervised learning?",
    options: {
      A: "Learning with labels",
      B: "Learning without labels",
      C: "Learning with rewards",
      D: "Learning from feedback",
    },
    correctAnswer: "B",
  },
  {
    id: 663,
    question: "What is reinforcement learning?",
    options: {
      A: "Learning from labeled data",
      B: "Learning from unlabeled data",
      C: "Learning from reward and punishment",
      D: "Learning from rules",
    },
    correctAnswer: "C",
  },
  {
    id: 664,
    question: "What is a dataset?",
    options: {
      A: "A model",
      B: "A collection of data",
      C: "A loss function",
      D: "An optimizer",
    },
    correctAnswer: "B",
  },
  {
    id: 665,
    question: "What is a feature?",
    options: {
      A: "A model",
      B: "An output",
      C: "An input variable",
      D: "A prediction",
    },
    correctAnswer: "C",
  },
  {
    id: 666,
    question: "What is a label?",
    options: {
      A: "Input variable",
      B: "Target output",
      C: "Feature name",
      D: "Dataset name",
    },
    correctAnswer: "B",
  },
  {
    id: 667,
    question: "Which task predicts continuous values?",
    options: {
      A: "Classification",
      B: "Clustering",
      C: "Regression",
      D: "Detection",
    },
    correctAnswer: "C",
  },
  {
    id: 668,
    question: "Which task predicts categories?",
    options: {
      A: "Regression",
      B: "Classification",
      C: "Clustering",
      D: "Forecasting",
    },
    correctAnswer: "B",
  },
  {
    id: 669,
    question: "Which algorithm is tree-based?",
    options: {
      A: "KNN",
      B: "Naive Bayes",
      C: "Decision Tree",
      D: "SVM",
    },
    correctAnswer: "C",
  },
  {
    id: 670,
    question: "Which ensemble method uses multiple trees?",
    options: {
      A: "Linear Regression",
      B: "Random Forest",
      C: "KNN",
      D: "Naive Bayes",
    },
    correctAnswer: "B",
  },
  {
    id: 671,
    question: "Which boosting algorithm is popular?",
    options: {
      A: "K-Means",
      B: "XGBoost",
      C: "PCA",
      D: "KNN",
    },
    correctAnswer: "B",
  },
  {
    id: 672,
    question: "What is ensemble learning?",
    options: {
      A: "Single model training",
      B: "Combining multiple models",
      C: "Feature engineering",
      D: "Model deployment",
    },
    correctAnswer: "B",
  },
  {
    id: 673,
    question: "What is bagging?",
    options: {
      A: "Boosting technique",
      B: "Feature selection",
      C: "Training models on random subsets",
      D: "Hyperparameter tuning",
    },
    correctAnswer: "C",
  },
  {
    id: 674,
    question: "What is boosting?",
    options: {
      A: "Training independent models",
      B: "Training models sequentially",
      C: "Reducing data size",
      D: "Removing noise",
    },
    correctAnswer: "B",
  },
  {
    id: 675,
    question: "What is stacking?",
    options: {
      A: "Combining models using meta-model",
      B: "Training data",
      C: "Feature extraction",
      D: "Model compression",
    },
    correctAnswer: "A",
  },
  {
    id: 676,
    question: "Which evaluation metric is used for imbalanced data?",
    options: {
      A: "Accuracy",
      B: "Precision-Recall",
      C: "MSE",
      D: "R2",
    },
    correctAnswer: "B",
  },
  {
    id: 677,
    question: "What is ROC curve?",
    options: {
      A: "Training curve",
      B: "Loss curve",
      C: "Performance curve",
      D: "Learning curve",
    },
    correctAnswer: "C",
  },
  {
    id: 678,
    question: "What is AUC?",
    options: {
      A: "Area under curve",
      B: "Accuracy unit",
      C: "Average user count",
      D: "Activation unit",
    },
    correctAnswer: "A",
  },
  {
    id: 679,
    question: "What is class imbalance?",
    options: {
      A: "Equal classes",
      B: "Unequal class distribution",
      C: "Missing data",
      D: "Noisy data",
    },
    correctAnswer: "B",
  },
  {
    id: 680,
    question: "Which technique handles imbalance?",
    options: {
      A: "Normalization",
      B: "SMOTE",
      C: "PCA",
      D: "Pruning",
    },
    correctAnswer: "B",
  },
  {
    id: 681,
    question: "What is SMOTE?",
    options: {
      A: "Feature scaling",
      B: "Oversampling technique",
      C: "Loss function",
      D: "Optimizer",
    },
    correctAnswer: "B",
  },
  {
    id: 682,
    question: "What is data augmentation?",
    options: {
      A: "Reducing dataset",
      B: "Generating new data from existing data",
      C: "Removing noise",
      D: "Splitting dataset",
    },
    correctAnswer: "B",
  },
  {
    id: 683,
    question: "Which is data augmentation for images?",
    options: {
      A: "Tokenization",
      B: "Rotation",
      C: "Stemming",
      D: "Stopword removal",
    },
    correctAnswer: "B",
  },
  {
    id: 684,
    question: "What is regularization?",
    options: {
      A: "Increasing model complexity",
      B: "Reducing overfitting",
      C: "Increasing dataset",
      D: "Reducing training time",
    },
    correctAnswer: "B",
  },
  {
    id: 685,
    question: "Which regularization adds penalty on weights?",
    options: {
      A: "Dropout",
      B: "L2 Regularization",
      C: "Early stopping",
      D: "Batch normalization",
    },
    correctAnswer: "B",
  },
  {
    id: 686,
    question: "What is batch normalization?",
    options: {
      A: "Scaling outputs",
      B: "Normalizing inputs per batch",
      C: "Reducing dataset",
      D: "Data cleaning",
    },
    correctAnswer: "B",
  },
  {
    id: 687,
    question: "What is vanishing gradient?",
    options: {
      A: "Gradient becomes too large",
      B: "Gradient becomes too small",
      C: "Loss becomes zero",
      D: "Accuracy becomes zero",
    },
    correctAnswer: "B",
  },
  {
    id: 688,
    question: "Which activation helps reduce vanishing gradient?",
    options: {
      A: "Sigmoid",
      B: "Tanh",
      C: "ReLU",
      D: "Softmax",
    },
    correctAnswer: "C",
  },
  {
    id: 689,
    question: "What is exploding gradient?",
    options: {
      A: "Gradient becomes very large",
      B: "Gradient becomes zero",
      C: "Loss becomes zero",
      D: "Accuracy increases",
    },
    correctAnswer: "A",
  },
  {
    id: 690,
    question: "Which technique handles exploding gradient?",
    options: {
      A: "Dropout",
      B: "Gradient clipping",
      C: "Normalization",
      D: "Regularization",
    },
    correctAnswer: "B",
  },
  {
    id: 691,
    question: "What is attention mechanism?",
    options: {
      A: "Feature scaling",
      B: "Focusing on important inputs",
      C: "Reducing model size",
      D: "Data preprocessing",
    },
    correctAnswer: "B",
  },
  {
    id: 692,
    question: "Which architecture uses attention?",
    options: {
      A: "CNN",
      B: "RNN",
      C: "Transformer",
      D: "Autoencoder",
    },
    correctAnswer: "C",
  },
  {
    id: 693,
    question: "What is self-attention?",
    options: {
      A: "Attention to other models",
      B: "Attention within same sequence",
      C: "Attention to images",
      D: "Attention to labels",
    },
    correctAnswer: "B",
  },
  {
    id: 694,
    question: "What is positional encoding?",
    options: {
      A: "Encoding word meaning",
      B: "Encoding word order",
      C: "Encoding labels",
      D: "Encoding images",
    },
    correctAnswer: "B",
  },
  {
    id: 695,
    question: "What is BERT?",
    options: {
      A: "Image model",
      B: "Text embedding model",
      C: "Clustering model",
      D: "Regression model",
    },
    correctAnswer: "B",
  },
  {
    id: 696,
    question: "What is GPT?",
    options: {
      A: "Discriminative model",
      B: "Generative language model",
      C: "Clustering model",
      D: "Vision model",
    },
    correctAnswer: "B",
  },
  {
    id: 697,
    question: "What is pretraining?",
    options: {
      A: "Training on small dataset",
      B: "Training before deployment",
      C: "Training on large generic dataset",
      D: "Training on test data",
    },
    correctAnswer: "C",
  },
  {
    id: 698,
    question: "What is fine-tuning in LLMs?",
    options: {
      A: "Training from scratch",
      B: "Adapting pretrained model",
      C: "Reducing model size",
      D: "Removing layers",
    },
    correctAnswer: "B",
  },
  {
    id: 699,
    question: "What is RLHF?",
    options: {
      A: "Reinforcement Learning with Human Feedback",
      B: "Random Learning Hyper Framework",
      C: "Recurrent Learning High Frequency",
      D: "Rule Learning Hybrid Flow",
    },
    correctAnswer: "A",
  },
  {
    id: 700,
    question: "Why is RLHF used?",
    options: {
      A: "To speed training",
      B: "To improve model alignment",
      C: "To reduce size",
      D: "To compress model",
    },
    correctAnswer: "B",
  },
  {
    id: 701,
    question: "What is alignment in AI?",
    options: {
      A: "Model speed",
      B: "Model following human intent",
      C: "Model accuracy",
      D: "Model size",
    },
    correctAnswer: "B",
  },
  {
    id: 702,
    question: "What is safety in AI?",
    options: {
      A: "Fast inference",
      B: "Preventing harmful outputs",
      C: "High accuracy",
      D: "Low latency",
    },
    correctAnswer: "B",
  },
  {
    id: 703,
    question: "What is responsible AI?",
    options: {
      A: "Fast AI",
      B: "Cheap AI",
      C: "Ethical and safe AI",
      D: "Open-source AI",
    },
    correctAnswer: "C",
  },
  {
    id: 704,
    question: "What is data privacy?",
    options: {
      A: "Public data",
      B: "Protecting user data",
      C: "Open datasets",
      D: "Model deployment",
    },
    correctAnswer: "B",
  },
  {
    id: 705,
    question: "What is GDPR?",
    options: {
      A: "AI model",
      B: "Data protection regulation",
      C: "Programming language",
      D: "Cloud platform",
    },
    correctAnswer: "B",
  },
  {
    id: 706,
    question: "What is model governance?",
    options: {
      A: "Model training",
      B: "Model monitoring and control",
      C: "Model architecture",
      D: "Model optimization",
    },
    correctAnswer: "B",
  },
  {
    id: 707,
    question: "What is audit trail?",
    options: {
      A: "Training log",
      B: "Prediction history",
      C: "Record of model decisions",
      D: "Loss history",
    },
    correctAnswer: "C",
  },
  {
    id: 708,
    question: "What is model versioning?",
    options: {
      A: "Changing dataset",
      B: "Tracking model changes",
      C: "Training model",
      D: "Deploying model",
    },
    correctAnswer: "B",
  },
  {
    id: 709,
    question: "What is experiment tracking?",
    options: {
      A: "Tracking users",
      B: "Tracking model runs",
      C: "Tracking predictions",
      D: "Tracking errors",
    },
    correctAnswer: "B",
  },
  {
    id: 710,
    question: "Which tool is used for experiment tracking?",
    options: {
      A: "MLflow",
      B: "Docker",
      C: "FastAPI",
      D: "Kafka",
    },
    correctAnswer: "A",
  },
  {
    id: 711,
    question: "What is containerization?",
    options: {
      A: "Model training",
      B: "Packaging application with dependencies",
      C: "Feature engineering",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 712,
    question: "Which tool is used for containerization?",
    options: {
      A: "Git",
      B: "Docker",
      C: "Kubernetes",
      D: "MLflow",
    },
    correctAnswer: "B",
  },
  {
    id: 713,
    question: "What is Kubernetes?",
    options: {
      A: "Database",
      B: "Model",
      C: "Container orchestration platform",
      D: "Programming language",
    },
    correctAnswer: "C",
  },
  {
    id: 714,
    question: "What is CI/CD?",
    options: {
      A: "Model training",
      B: "Continuous integration and deployment",
      C: "Feature scaling",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 715,
    question: "What is API?",
    options: {
      A: "Model",
      B: "Interface for software communication",
      C: "Dataset",
      D: "Framework",
    },
    correctAnswer: "B",
  },
  {
    id: 716,
    question: "Which framework is used for model serving?",
    options: {
      A: "FastAPI",
      B: "NumPy",
      C: "Pandas",
      D: "Matplotlib",
    },
    correctAnswer: "A",
  },
  {
    id: 717,
    question: "What is REST API?",
    options: {
      A: "Database",
      B: "Model",
      C: "Web service architecture",
      D: "Programming language",
    },
    correctAnswer: "C",
  },
  {
    id: 718,
    question: "What is latency?",
    options: {
      A: "Training time",
      B: "Inference time",
      C: "Deployment time",
      D: "Data loading time",
    },
    correctAnswer: "B",
  },
  {
    id: 719,
    question: "What is throughput?",
    options: {
      A: "Requests per second",
      B: "Training speed",
      C: "Model size",
      D: "Accuracy",
    },
    correctAnswer: "A",
  },
  {
    id: 720,
    question: "What is scalability?",
    options: {
      A: "Model size",
      B: "Ability to handle increased load",
      C: "Training speed",
      D: "Accuracy",
    },
    correctAnswer: "B",
  },
  {
    id: 721,
    question: "What is cloud computing?",
    options: {
      A: "Local storage",
      B: "On-demand computing resources",
      C: "Offline computing",
      D: "Edge computing",
    },
    correctAnswer: "B",
  },
  {
    id: 722,
    question: "Which is cloud provider?",
    options: {
      A: "AWS",
      B: "TensorFlow",
      C: "Docker",
      D: "Git",
    },
    correctAnswer: "A",
  },
  {
    id: 723,
    question: "What is GPU used for?",
    options: {
      A: "Data storage",
      B: "Parallel computation",
      C: "Networking",
      D: "Logging",
    },
    correctAnswer: "B",
  },
  {
    id: 724,
    question: "Why GPUs are used in AI?",
    options: {
      A: "They are cheaper",
      B: "They consume less power",
      C: "They accelerate matrix operations",
      D: "They store more data",
    },
    correctAnswer: "C",
  },
  {
    id: 725,
    question: "What is TPU?",
    options: {
      A: "Training Processing Unit",
      B: "Tensor Processing Unit",
      C: "Text Processing Unit",
      D: "Transfer Processing Unit",
    },
    correctAnswer: "B",
  },
  {
    id: 726,
    question: "What is edge deployment?",
    options: {
      A: "Deploying on cloud",
      B: "Deploying on local devices",
      C: "Deploying on servers",
      D: "Deploying on database",
    },
    correctAnswer: "B",
  },
  {
    id: 727,
    question: "What is serverless computing?",
    options: {
      A: "No servers used",
      B: "Managed backend services",
      C: "Local deployment",
      D: "Edge deployment",
    },
    correctAnswer: "B",
  },
  {
    id: 728,
    question: "Which is serverless platform?",
    options: {
      A: "AWS Lambda",
      B: "Docker",
      C: "Kubernetes",
      D: "FastAPI",
    },
    correctAnswer: "A",
  },
  {
    id: 729,
    question: "What is cost optimization?",
    options: {
      A: "Increasing compute",
      B: "Reducing infrastructure cost",
      C: "Increasing model size",
      D: "Increasing accuracy",
    },
    correctAnswer: "B",
  },
  {
    id: 730,
    question: "What is autoscaling?",
    options: {
      A: "Manual scaling",
      B: "Automatic resource scaling",
      C: "Model scaling",
      D: "Data scaling",
    },
    correctAnswer: "B",
  },
  {
    id: 731,
    question: "What is load balancing?",
    options: {
      A: "Balancing dataset",
      B: "Distributing traffic across servers",
      C: "Training model",
      D: "Optimizing features",
    },
    correctAnswer: "B",
  },
  {
    id: 732,
    question: "What is monitoring?",
    options: {
      A: "Training model",
      B: "Tracking system performance",
      C: "Data preprocessing",
      D: "Feature engineering",
    },
    correctAnswer: "B",
  },
  {
    id: 733,
    question: "What is logging?",
    options: {
      A: "Tracking events and errors",
      B: "Training model",
      C: "Deploying model",
      D: "Scaling system",
    },
    correctAnswer: "A",
  },
  {
    id: 734,
    question: "What is alerting?",
    options: {
      A: "Sending notifications on issues",
      B: "Training model",
      C: "Deploying model",
      D: "Scaling model",
    },
    correctAnswer: "A",
  },
  {
    id: 735,
    question: "What is rollback?",
    options: {
      A: "Deploy new version",
      B: "Revert to previous version",
      C: "Train model",
      D: "Test model",
    },
    correctAnswer: "B",
  },
  {
    id: 736,
    question: "What is canary deployment?",
    options: {
      A: "Deploy to all users",
      B: "Deploy to small group first",
      C: "Shadow deployment",
      D: "Offline deployment",
    },
    correctAnswer: "B",
  },
  {
    id: 737,
    question: "What is blue-green deployment?",
    options: {
      A: "Two environments switching",
      B: "Training technique",
      C: "Data pipeline",
      D: "Feature selection",
    },
    correctAnswer: "A",
  },
  {
    id: 738,
    question: "What is uptime?",
    options: {
      A: "Training time",
      B: "System availability",
      C: "Deployment time",
      D: "Inference time",
    },
    correctAnswer: "B",
  },
  {
    id: 739,
    question: "What is SLA?",
    options: {
      A: "Service Level Agreement",
      B: "System Load Average",
      C: "Server Logging Agent",
      D: "Software License Agreement",
    },
    correctAnswer: "A",
  },
  {
    id: 740,
    question: "What is fault tolerance?",
    options: {
      A: "System speed",
      B: "System reliability",
      C: "System ability to continue on failure",
      D: "System accuracy",
    },
    correctAnswer: "C",
  },
  {
    id: 741,
    question: "What is redundancy?",
    options: {
      A: "Extra components for reliability",
      B: "Extra features",
      C: "Extra data",
      D: "Extra models",
    },
    correctAnswer: "A",
  },
  {
    id: 742,
    question: "What is backup?",
    options: {
      A: "Copy of data",
      B: "Model training",
      C: "Feature scaling",
      D: "Deployment",
    },
    correctAnswer: "A",
  },
  {
    id: 743,
    question: "What is disaster recovery?",
    options: {
      A: "Training model",
      B: "Recovering system after failure",
      C: "Feature engineering",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 744,
    question: "What is business continuity?",
    options: {
      A: "System scaling",
      B: "Ensuring operations during disruption",
      C: "Model training",
      D: "Data preprocessing",
    },
    correctAnswer: "B",
  },
  {
    id: 745,
    question: "What is cybersecurity?",
    options: {
      A: "Model training",
      B: "Protecting systems and data",
      C: "Feature engineering",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 746,
    question: "What is encryption?",
    options: {
      A: "Data compression",
      B: "Securing data",
      C: "Data cleaning",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 747,
    question: "What is authentication?",
    options: {
      A: "Data encryption",
      B: "User verification",
      C: "System monitoring",
      D: "Access logging",
    },
    correctAnswer: "B",
  },
  {
    id: 748,
    question: "What is authorization?",
    options: {
      A: "User verification",
      B: "Access control",
      C: "Data encryption",
      D: "Logging",
    },
    correctAnswer: "B",
  },
  {
    id: 749,
    question: "What is role-based access control?",
    options: {
      A: "Access based on roles",
      B: "Access based on IP",
      C: "Access based on password",
      D: "Access based on token",
    },
    correctAnswer: "A",
  },
  {
    id: 750,
    question: "What is token-based authentication?",
    options: {
      A: "Using username/password",
      B: "Using session cookies",
      C: "Using access tokens",
      D: "Using biometrics",
    },
    correctAnswer: "C",
  },
  {
    id: 751,
    question: "What is OAuth?",
    options: {
      A: "Database",
      B: "Authentication protocol",
      C: "Programming language",
      D: "Cloud service",
    },
    correctAnswer: "B",
  },
  {
    id: 752,
    question: "What is API security?",
    options: {
      A: "Model training",
      B: "Protecting APIs from attacks",
      C: "Feature engineering",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 753,
    question: "What is rate limiting?",
    options: {
      A: "Limiting training speed",
      B: "Limiting API requests",
      C: "Limiting dataset size",
      D: "Limiting model size",
    },
    correctAnswer: "B",
  },
  {
    id: 754,
    question: "What is throttling?",
    options: {
      A: "Blocking users",
      B: "Limiting request speed",
      C: "Encrypting data",
      D: "Logging requests",
    },
    correctAnswer: "B",
  },
  {
    id: 755,
    question: "What is firewall?",
    options: {
      A: "Security system",
      B: "Database",
      C: "Model",
      D: "Framework",
    },
    correctAnswer: "A",
  },
  {
    id: 756,
    question: "What is intrusion detection?",
    options: {
      A: "Monitoring attacks",
      B: "Training model",
      C: "Feature engineering",
      D: "Data labeling",
    },
    correctAnswer: "A",
  },
  {
    id: 757,
    question: "What is penetration testing?",
    options: {
      A: "Testing model accuracy",
      B: "Testing system security",
      C: "Testing performance",
      D: "Testing deployment",
    },
    correctAnswer: "B",
  },
  {
    id: 758,
    question: "What is vulnerability?",
    options: {
      A: "System weakness",
      B: "System strength",
      C: "System speed",
      D: "System accuracy",
    },
    correctAnswer: "A",
  },
  {
    id: 759,
    question: "What is patching?",
    options: {
      A: "Fixing security issues",
      B: "Training model",
      C: "Deploying model",
      D: "Scaling system",
    },
    correctAnswer: "A",
  },
  {
    id: 760,
    question: "What is compliance?",
    options: {
      A: "Model training",
      B: "Following regulations and standards",
      C: "Feature engineering",
      D: "Data labeling",
    },
    correctAnswer: "B",
  },
  {
    id: 761,
    question: "What is AWS SageMaker?",
    options: {
      A: "A cloud database service",
      B: "A cloud platform for building, training, and deploying ML models",
      C: "A data visualization tool",
      D: "A serverless compute service",
    },
    correctAnswer: "B",
  },
  {
    id: 762,
    question: "AWS SageMaker is mainly used for:",
    options: {
      A: "Web hosting",
      B: "Machine learning lifecycle management",
      C: "Database backup",
      D: "Logging and monitoring",
    },
    correctAnswer: "B",
  },
  {
    id: 763,
    question: "Which feature of SageMaker helps in model deployment?",
    options: {
      A: "EC2",
      B: "S3",
      C: "SageMaker Endpoints",
      D: "Lambda",
    },
    correctAnswer: "C",
  },
  {
    id: 764,
    question: "Azure AI is a platform provided by:",
    options: {
      A: "Google",
      B: "Amazon",
      C: "Microsoft",
      D: "IBM",
    },
    correctAnswer: "C",
  },
  {
    id: 765,
    question: "Which Azure service is used for building ML models?",
    options: {
      A: "Azure Blob Storage",
      B: "Azure DevOps",
      C: "Azure Machine Learning",
      D: "Azure SQL",
    },
    correctAnswer: "C",
  },
  {
    id: 766,
    question: "Azure AI Cognitive Services provides:",
    options: {
      A: "Only storage",
      B: "Pre-built AI models for vision, speech, and language",
      C: "Only GPU machines",
      D: "Only ML training",
    },
    correctAnswer: "B",
  },
  {
    id: 767,
    question: "Google Vertex AI is:",
    options: {
      A: "A database service",
      B: "A cloud AI platform for ML model lifecycle",
      C: "A networking service",
      D: "A DevOps tool",
    },
    correctAnswer: "B",
  },
  {
    id: 768,
    question: "Vertex AI is provided by:",
    options: {
      A: "Amazon",
      B: "Microsoft",
      C: "Google",
      D: "Meta",
    },
    correctAnswer: "C",
  },
  {
    id: 769,
    question:
      "Which service is common across SageMaker, Azure ML, and Vertex AI?",
    options: {
      A: "Web hosting",
      B: "Model training and deployment",
      C: "Database management",
      D: "Email service",
    },
    correctAnswer: "B",
  },
  {
    id: 770,
    question: "Cloud AI platforms are mainly used to:",
    options: {
      A: "Replace data scientists",
      B: "Provide scalable ML infrastructure",
      C: "Store only datasets",
      D: "Run only web apps",
    },
    correctAnswer: "B",
  },
  {
    id: 771,
    question: "Which is an advantage of cloud AI platforms?",
    options: {
      A: "Limited scalability",
      B: "Manual infrastructure setup",
      C: "On-demand compute and scaling",
      D: "No GPU support",
    },
    correctAnswer: "C",
  },
  {
    id: 772,
    question: "Discriminative models mainly focus on:",
    options: {
      A: "Generating new data",
      B: "Learning joint probability of data",
      C: "Learning decision boundaries between classes",
      D: "Creating images",
    },
    correctAnswer: "C",
  },
  {
    id: 773,
    question: "Which is a discriminative model?",
    options: {
      A: "GAN",
      B: "Naive Bayes",
      C: "Logistic Regression",
      D: "Diffusion model",
    },
    correctAnswer: "C",
  },
  {
    id: 774,
    question: "Generative models mainly focus on:",
    options: {
      A: "Predicting labels",
      B: "Generating new data samples",
      C: "Finding decision boundaries",
      D: "Feature scaling",
    },
    correctAnswer: "B",
  },
  {
    id: 775,
    question: "Which is a generative model?",
    options: {
      A: "SVM",
      B: "Random Forest",
      C: "GAN",
      D: "Logistic Regression",
    },
    correctAnswer: "C",
  },
  {
    id: 776,
    question: "Predictive AI is mainly used for:",
    options: {
      A: "Generating content",
      B: "Making future predictions",
      C: "Recommending actions",
      D: "Automating workflows",
    },
    correctAnswer: "B",
  },
  {
    id: 777,
    question: "Which is an example of Predictive AI?",
    options: {
      A: "Chatbot",
      B: "Fraud prediction model",
      C: "Image generator",
      D: "Music generation model",
    },
    correctAnswer: "B",
  },
  {
    id: 778,
    question: "Prescriptive AI is mainly used for:",
    options: {
      A: "Predicting future outcomes",
      B: "Describing past data",
      C: "Recommending best actions to take",
      D: "Generating text",
    },
    correctAnswer: "C",
  },
  {
    id: 779,
    question: "Which is an example of Prescriptive AI?",
    options: {
      A: "Sales forecasting model",
      B: "Recommendation engine suggesting next product",
      C: "Spam classifier",
      D: "Image classifier",
    },
    correctAnswer: "B",
  },
  {
    id: 780,
    question: "Chatbots are primarily an application of:",
    options: {
      A: "Computer Vision",
      B: "Reinforcement Learning",
      C: "Natural Language Processing",
      D: "Time-series analysis",
    },
    correctAnswer: "C",
  },
  {
    id: 781,
    question: "Which AI model is commonly used in chatbots?",
    options: {
      A: "CNN",
      B: "LLM",
      C: "K-Means",
      D: "PCA",
    },
    correctAnswer: "B",
  },
  {
    id: 782,
    question: "A customer support chatbot is an example of:",
    options: {
      A: "Predictive AI",
      B: "Generative AI",
      C: "Descriptive AI",
      D: "Clustering AI",
    },
    correctAnswer: "B",
  },
  {
    id: 783,
    question: "Recommendation engines are mainly used to:",
    options: {
      A: "Detect fraud",
      B: "Translate languages",
      C: "Suggest relevant items to users",
      D: "Generate images",
    },
    correctAnswer: "C",
  },
  {
    id: 784,
    question: "Netflix and Amazon product suggestions are examples of:",
    options: {
      A: "Chatbots",
      B: "Recommendation systems",
      C: "OCR systems",
      D: "Fraud detection systems",
    },
    correctAnswer: "B",
  },
  {
    id: 785,
    question: "Which technique is widely used in recommendation systems?",
    options: {
      A: "Clustering",
      B: "Collaborative filtering",
      C: "Decision trees",
      D: "Linear regression",
    },
    correctAnswer: "B",
  },
  {
    id: 786,
    question: "Fraud detection systems are mainly used to:",
    options: {
      A: "Generate fake transactions",
      B: "Predict future sales",
      C: "Detect abnormal or suspicious activity",
      D: "Improve customer support",
    },
    correctAnswer: "C",
  },
  {
    id: 787,
    question: "Credit card fraud detection is an example of:",
    options: {
      A: "Recommendation system",
      B: "Chatbot",
      C: "Anomaly detection",
      D: "OCR",
    },
    correctAnswer: "C",
  },
  {
    id: 788,
    question: "OCR stands for:",
    options: {
      A: "Object Classification Rule",
      B: "Optical Character Recognition",
      C: "Online Content Retrieval",
      D: "Open Code Repository",
    },
    correctAnswer: "B",
  },
  {
    id: 789,
    question: "OCR is mainly used for:",
    options: {
      A: "Speech recognition",
      B: "Image generation",
      C: "Extracting text from images or scanned documents",
      D: "Fraud detection",
    },
    correctAnswer: "C",
  },
  {
    id: 790,
    question: "Which is a real-world OCR application?",
    options: {
      A: "Face recognition",
      B: "Document digitization",
      C: "Recommendation engine",
      D: "Chatbot",
    },
    correctAnswer: "B",
  },
  {
    id: 791,
    question: "Speech recognition is used to:",
    options: {
      A: "Generate music",
      B: "Convert text to audio",
      C: "Convert speech into text",
      D: "Translate images",
    },
    correctAnswer: "C",
  },
  {
    id: 792,
    question: "Which is a speech recognition application?",
    options: {
      A: "ChatGPT",
      B: "Google Assistant voice input",
      C: "Netflix recommendations",
      D: "OCR scanner",
    },
    correctAnswer: "B",
  },
  {
    id: 793,
    question: "Text-to-Speech (TTS) systems are used to:",
    options: {
      A: "Convert speech to text",
      B: "Convert text into spoken audio",
      C: "Detect emotions",
      D: "Summarize text",
    },
    correctAnswer: "B",
  },
  {
    id: 794,
    question: "Which AI field powers speech recognition and chatbots?",
    options: {
      A: "Computer Vision",
      B: "Robotics",
      C: "Natural Language Processing",
      D: "Data Mining",
    },
    correctAnswer: "C",
  },
  {
    id: 795,
    question: "Which of the following is a predictive AI use-case?",
    options: {
      A: "Movie recommendation",
      B: "Customer churn prediction",
      C: "Chatbot conversation",
      D: "Image generation",
    },
    correctAnswer: "B",
  },
  {
    id: 796,
    question: "Which of the following is a prescriptive AI use-case?",
    options: {
      A: "Weather forecasting",
      B: "Stock price prediction",
      C: "Suggesting best marketing campaign strategy",
      D: "Image classification",
    },
    correctAnswer: "C",
  },
  {
    id: 797,
    question:
      "Which real-world system often uses both predictive and prescriptive AI?",
    options: {
      A: "OCR scanner",
      B: "Recommendation engine",
      C: "Image classifier",
      D: "Speech recognition system",
    },
    correctAnswer: "B",
  },
  {
    id: 798,
    question:
      "Which AI application is used in call centers for voice automation?",
    options: {
      A: "OCR",
      B: "Fraud detection",
      C: "Speech recognition",
      D: "Image segmentation",
    },
    correctAnswer: "C",
  },
  {
    id: 799,
    question:
      "Which AI system is used to automatically read invoices and forms?",
    options: {
      A: "Chatbot",
      B: "Recommendation engine",
      C: "OCR",
      D: "Fraud detection",
    },
    correctAnswer: "C",
  },
  {
    id: 800,
    question: "Which AI system helps banks detect suspicious transactions?",
    options: {
      A: "Chatbot",
      B: "OCR",
      C: "Fraud detection system",
      D: "Recommendation engine",
    },
    correctAnswer: "C",
  },
  {
    id: 801,
    question: "Which AI platform would you use to train models on AWS cloud?",
    options: {
      A: "Vertex AI",
      B: "Azure AI",
      C: "AWS SageMaker",
      D: "TensorFlow",
    },
    correctAnswer: "C",
  },
  {
    id: 802,
    question:
      "Which cloud platform is best suited if your infrastructure is on Azure?",
    options: {
      A: "Vertex AI",
      B: "AWS SageMaker",
      C: "Azure Machine Learning",
      D: "Chroma",
    },
    correctAnswer: "C",
  },
  {
    id: 803,
    question: "Which platform integrates best with Google Cloud ecosystem?",
    options: {
      A: "AWS SageMaker",
      B: "Azure AI",
      C: "Vertex AI",
      D: "FAISS",
    },
    correctAnswer: "C",
  },
  {
    id: 804,
    question:
      "Which system is mainly responsible for personalized product suggestions?",
    options: {
      A: "OCR",
      B: "Chatbot",
      C: "Recommendation engine",
      D: "Fraud detection",
    },
    correctAnswer: "C",
  },
  {
    id: 805,
    question:
      "Which AI solution improves customer experience through automated conversations?",
    options: {
      A: "OCR",
      B: "Fraud detection",
      C: "Chatbot",
      D: "Speech recognition",
    },
    correctAnswer: "C",
  },
  {
    id: 806,
    question: "Which is an example of generative AI use-case?",
    options: {
      A: "Customer segmentation",
      B: "Image generation",
      C: "Fraud detection",
      D: "OCR",
    },
    correctAnswer: "B",
  },
  {
    id: 807,
    question: "Which is an example of discriminative AI use-case?",
    options: {
      A: "Text generation",
      B: "Image generation",
      C: "Spam classification",
      D: "Music generation",
    },
    correctAnswer: "C",
  },
  {
    id: 808,
    question: "Which AI system converts handwritten notes into editable text?",
    options: {
      A: "Speech recognition",
      B: "Chatbot",
      C: "OCR",
      D: "Recommendation engine",
    },
    correctAnswer: "C",
  },
  {
    id: 809,
    question: "Which AI application helps e-commerce platforms increase sales?",
    options: {
      A: "OCR",
      B: "Recommendation engine",
      C: "Fraud detection",
      D: "Speech recognition",
    },
    correctAnswer: "B",
  },
  {
    id: 810,
    question: "Which AI technology enables voice-based virtual assistants?",
    options: {
      A: "OCR",
      B: "Chatbot",
      C: "Speech recognition",
      D: "Fraud detection",
    },
    correctAnswer: "C",
  },
  {
    id: 811,
    question: "Which AI type focuses on learning the boundary between classes?",
    options: {
      A: "Generative",
      B: "Discriminative",
      C: "Prescriptive",
      D: "Descriptive",
    },
    correctAnswer: "B",
  },
  {
    id: 812,
    question: "Which AI type focuses on creating new samples?",
    options: {
      A: "Predictive",
      B: "Prescriptive",
      C: "Discriminative",
      D: "Generative",
    },
    correctAnswer: "D",
  },
  {
    id: 813,
    question: "Which AI type answers: 'What will happen?'",
    options: {
      A: "Descriptive",
      B: "Predictive",
      C: "Prescriptive",
      D: "Generative",
    },
    correctAnswer: "B",
  },
  {
    id: 814,
    question: "Which AI type answers: 'What should we do?'",
    options: {
      A: "Predictive",
      B: "Descriptive",
      C: "Prescriptive",
      D: "Discriminative",
    },
    correctAnswer: "C",
  },
  {
    id: 815,
    question: "Which AI type answers: 'What happened?'",
    options: {
      A: "Descriptive",
      B: "Predictive",
      C: "Prescriptive",
      D: "Generative",
    },
    correctAnswer: "A",
  },
  {
    id: 816,
    question: "Which real-world system uses both NLP and Generative AI?",
    options: {
      A: "OCR",
      B: "Chatbot",
      C: "Fraud detection",
      D: "Recommendation engine",
    },
    correctAnswer: "B",
  },
  {
    id: 817,
    question: "Which AI solution is most useful for digitizing old books?",
    options: {
      A: "Chatbot",
      B: "Speech recognition",
      C: "OCR",
      D: "Fraud detection",
    },
    correctAnswer: "C",
  },
  {
    id: 818,
    question: "Which AI system is widely used in banking and finance?",
    options: {
      A: "Chatbot",
      B: "Fraud detection",
      C: "OCR",
      D: "Speech recognition",
    },
    correctAnswer: "B",
  },
  {
    id: 819,
    question: "Which AI system helps platforms personalize user experience?",
    options: {
      A: "OCR",
      B: "Chatbot",
      C: "Recommendation engine",
      D: "Speech recognition",
    },
    correctAnswer: "C",
  },
  {
    id: 820,
    question: "Which combination is most common in modern AI products?",
    options: {
      A: "OCR + Fraud Detection",
      B: "Chatbot + LLM",
      C: "Recommendation + OCR",
      D: "Speech + Fraud Detection",
    },
    correctAnswer: "B",
  },
];

