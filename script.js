let selectedQuestions = [];
let currentIndex = 0;
let userAnswers = [];
let startTime;
let timerInterval;

function startTest() {
  // Hide Start & Result Screens
  document.getElementById("start-screen").style.display = "none";
  document.getElementById("result-screen").style.display = "none";

  // Show Quiz Screen
  document.getElementById("quiz-screen").style.display = "block";

  const selectedValue = document.getElementById("num-questions").value;

  if (selectedValue === "all") {
    selectedQuestions = shuffleArray([...questions]);
  } else {
    const num = parseInt(selectedValue);
    selectedQuestions = shuffleArray([...questions]).slice(0, num);
  }

  currentIndex = 0;
  userAnswers = [];

  startTimer();
  loadQuestion();
}

function startTimer() {
  startTime = Date.now();

  timerInterval = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);

    const hours = Math.floor(elapsedSeconds / 3600);
    const minutes = Math.floor((elapsedSeconds % 3600) / 60);
    const seconds = elapsedSeconds % 60;

    let timeText = "";

    if (hours > 0) {
      timeText = `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    } else {
      timeText = `${pad(minutes)}:${pad(seconds)}`;
    }

    document.getElementById("timer").innerText = `⏱ ${timeText}`;
  }, 1000);
}

function loadQuestion() {
  const q = selectedQuestions[currentIndex];
  document.getElementById("question-count").innerText = `Question ${
    currentIndex + 1
  } / ${selectedQuestions.length}`;

  document.getElementById("progress-bar").style.width =
    ((currentIndex + 1) / selectedQuestions.length) * 100 + "%";

  document.getElementById("question").innerText = q.question;

  const optionsDiv = document.getElementById("options");
  optionsDiv.innerHTML = "";

  for (let key in q.options) {
    const optionLabel = document.createElement("label");
    optionLabel.classList.add("option-label");
    optionLabel.innerHTML = `
      <input type="radio" name="option" value="${key}">
      ${key}. ${q.options[key]}
    `;
    optionLabel
      .querySelector("input")
      .addEventListener("click", () => selectOption(key));
    optionsDiv.appendChild(optionLabel);
  }

  // Disable Next button until selection
  document.querySelector(".btn.next").disabled = true;
}

// Handle option selection with feedback
function selectOption(selectedKey) {
  const q = selectedQuestions[currentIndex];
  userAnswers[currentIndex] = selectedKey;

  const inputs = document.querySelectorAll('input[name="option"]');
  inputs.forEach((input) => (input.disabled = true));

  inputs.forEach((input) => {
    const parent = input.parentElement;
    parent.classList.remove("correct", "wrong"); // Reset previous

    if (input.value === q.correctAnswer) {
      parent.classList.add("correct"); // Animate correct
    } else if (input.value === selectedKey && selectedKey !== q.correctAnswer) {
      parent.classList.add("wrong"); // Animate wrong
    }
  });

  // Enable Next button after selection
  document.querySelector(".btn.next").disabled = false;
}

function nextQuestion() {
  currentIndex++;
  if (currentIndex < selectedQuestions.length) {
    loadQuestion();
  } else {
    finishTest();
  }
}

function finishTest() {
  clearInterval(timerInterval);

  document.getElementById("quiz-screen").style.display = "none";
  document.getElementById("result-screen").style.display = "block";

  const totalSeconds = Math.floor((Date.now() - startTime) / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  let formattedTime = "";
  if (hours > 0) {
    formattedTime = `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
  } else {
    formattedTime = `${pad(minutes)}:${pad(seconds)}`;
  }

  let score = 0;
  let wrongHTML = "";
  let wrongCount = 0;

  selectedQuestions.forEach((q, i) => {
    if (userAnswers[i] === q.correctAnswer) {
      score++;
    } else {
      wrongCount++;
      wrongHTML += `
        <p>
          <strong>Q:</strong> ${q.question}<br>
      
          <strong>Your Answer:</strong>
          ${
            userAnswers[i] && q.options[userAnswers[i]]
              ? `${userAnswers[i]}. ${q.options[userAnswers[i]]}`
              : "Not Answered"
          }
          <br>
      
          <strong>Correct Answer:</strong>
          ${q.correctAnswer}. ${q.options[q.correctAnswer]}
        </p>
        <hr>
      `;
    }
  });

  document.getElementById(
    "time-taken"
  ).innerText = `Time Taken: ${formattedTime}`;

  document.getElementById(
    "score"
  ).innerText = `Score: ${score} / ${selectedQuestions.length}`;
  document.getElementById(
    "wrong-count"
  ).innerText = `Wrong Answers: ${wrongCount}`;
  document.getElementById("wrong-answers").innerHTML = wrongHTML;
}

function showNotes() {
  const dropdown = document.getElementById("notes-topics");
  const selectedOptions = Array.from(dropdown.selectedOptions);
  const output = document.getElementById("notes-output");

  if (selectedOptions.length === 0) {
    output.innerHTML = "<p>Please select at least one topic.</p>";
    output.classList.remove("hidden");
    return;
  }

  let notesHTML = `
    <h4 style="margin-bottom:15px;color:#2b2f77;">
      Topics You Should Revise
    </h4>
  `;

  selectedOptions.forEach(option => {
    notesHTML += getNotesForTopic(option.value);
  });

  output.innerHTML = notesHTML;
  output.classList.remove("hidden");
}



function getNotesForTopic(topic) {
  const notes = {
    ai_basics: `
      <div class="note-block">
        <h5>AI Basics</h5>
        <ul>
          <li>AI = Machines that mimic human intelligence</li>
          <li>ML = Machines learn patterns from data</li>
          <li>Deep Learning = Neural networks with many layers</li>
          <li>Examples: Chatbots, Face recognition, Recommendation systems</li>
          <li>Narrow AI = Task specific (most AI today)</li>
          <li>General AI = Human-level intelligence (future goal)</li>
        </ul>
      </div>
    `,

    ml_basics: `
      <div class="note-block">
        <h5>Machine Learning</h5>
        <ul>
          <li>Supervised Learning → Labeled data (Classification, Regression)</li>
          <li>Unsupervised Learning → No labels (Clustering, PCA)</li>
          <li>Reinforcement Learning → Reward & punishment</li>
          <li>Features = Input variables</li>
          <li>Labels = Target output</li>
          <li>Overfitting = Memorizing training data</li>
          <li>Underfitting = Too simple model</li>
        </ul>
      </div>
    `,

    deep_learning: `
      <div class="note-block">
        <h5>Deep Learning</h5>
        <ul>
          <li>Neural Network = Layers of neurons</li>
          <li>CNN → Image processing</li>
          <li>RNN / LSTM → Sequential data (text, speech)</li>
          <li>Transformer → Used in LLMs</li>
          <li>Epoch = One full dataset pass</li>
          <li>Batch = Subset of data</li>
          <li>Activation: ReLU, Sigmoid</li>
          <li>Loss: Cross Entropy (classification)</li>
          <li>Optimizer: Adam</li>
        </ul>
      </div>
    `,

    llm_genai: `
      <div class="note-block">
        <h5>LLM & Generative AI</h5>
        <ul>
          <li>LLM = Large Language Model</li>
          <li>Examples: GPT, BERT</li>
          <li>Token = Word or sub-word unit</li>
          <li>Context Window = Max input size</li>
          <li>Temperature = Randomness control</li>
          <li>Hallucination = Incorrect output</li>
          <li>RLHF = Reinforcement Learning with Human Feedback</li>
          <li>Generative AI = Generates new content (text, image, audio)</li>
        </ul>
      </div>
    `,

    prompting: `
      <div class="note-block">
        <h5>Prompt Engineering</h5>
        <ul>
          <li>Zero-shot → No examples</li>
          <li>Few-shot → Few examples</li>
          <li>Chain-of-thought → Step-by-step reasoning</li>
          <li>System prompt → Controls model behavior</li>
          <li>Good prompt = Clear + Specific + Structured</li>
          <li>Bad prompt = Vague, short, unclear</li>
        </ul>
      </div>
    `,

    rag: `
      <div class="note-block">
        <h5>RAG (Retrieval Augmented Generation)</h5>
        <ul>
          <li>Used to reduce hallucination</li>
          <li>Combines Vector DB + LLM</li>
          <li>Flow: Query → Embeddings → Vector DB → Retrieve → LLM</li>
          <li>Embeddings = Numerical representation of text</li>
          <li>Vector DB = FAISS, Pinecone, Chroma</li>
          <li>Chunking = Splitting documents</li>
        </ul>
      </div>
    `,

    cloud: `
      <div class="note-block">
        <h5>Cloud AI Platforms</h5>
        <ul>
          <li>AWS SageMaker → Amazon ML platform</li>
          <li>Azure AI / Azure ML → Microsoft ML platform</li>
          <li>Google Vertex AI → Google ML platform</li>
          <li>Used for: Training, Deployment, Scaling</li>
          <li>Provides GPU, TPU, Auto-scaling</li>
        </ul>
      </div>
    `,

    usecases: `
      <div class="note-block">
        <h5>Real-world AI Use-cases</h5>
        <ul>
          <li>Chatbots → NLP + LLM</li>
          <li>Recommendation Engines → ML</li>
          <li>Fraud Detection → Anomaly Detection</li>
          <li>OCR → Extract text from images</li>
          <li>Speech Recognition → Convert voice to text</li>
          <li>Face Recognition → Computer Vision</li>
        </ul>
      </div>
    `
  };

  return notes[topic] || "";
}



function shuffleArray(array) {
  return array.sort(() => Math.random() - 0.5);
}

function pad(num) {
  return num.toString().padStart(2, "0");
}



