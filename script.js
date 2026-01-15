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
          <strong>Your Answer:</strong> ${userAnswers[i] || "Not Answered"}<br>
          <strong>Correct Answer:</strong> ${q.correctAnswer}
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

function shuffleArray(array) {
  return array.sort(() => Math.random() - 0.5);
}

function pad(num) {
  return num.toString().padStart(2, "0");
}
