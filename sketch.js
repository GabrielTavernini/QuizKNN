/*
  KNN Classification took from https://www.youtube.com/watch?v=Mwo5_bUVhlA
*/
let video;
let features;
let knn;
let labelP;
let labelQuest;
let labelScore;
let ready = false;
let label = 'nothing';
let start = false;



class Question {
  constructor(quest, answ) {
    this.question = quest;
    this.answer = answ;
  }
}

let counter = 0;
let score = 0;
let questionIndex = -1;
let currentQuestion = new Question("Ready?", true);
let questions = [
  new Question("Mark Zuckerberg is human", false), 
  new Question("Tabs are better than spaces!", true), 
  new Question("AI is just a bunch of if statement", true)
];



function setup() {
  createCanvas(500, 400);
  textAlign(CENTER, CENTER);
  textSize(32);
  background(61);

  video = createCapture(VIDEO);
  video.size(200, 200);
  video.position(21, -5);
  features = ml5.featureExtractor('MobileNet', modelReady);
  knn = ml5.KNNClassifier();
}

function draw() {
  if (!ready && knn.getNumLabels() > 0) {
    goClassify();
    ready = true;
  }

  if (start) {
    if (counter > 150) {
      if (label == "true" && currentQuestion.answer)
        score++;

      if (label == "false" && !currentQuestion.answer)
        score++;

      questionIndex++;
      currentQuestion = questions[questionIndex];
      counter = 0;
    }

    if(questionIndex < questions.length) {
      if (label != "nothing")
        counter++;

      text((questionIndex + 1) + ". " + currentQuestion.question, width / 2, height / 2 + 80)
      text("Score: " + score, width - 140, 90)
    } else {
      video.hide();
      if(score > questions.length/2)
        background("#0fff4b");
      else
        background("#ff430f");

      text("Score: " + score, width/2, height/2);
    }
  } else {
    fill(255);
    text("Train the classifier!", width / 2, height / 2 + 70 - 16);
    text("Press S to start", width / 2, height / 2 + 70 + 16);
  }
}

function keyPressed() {
  const logits = features.infer(video);
  if (key == 'T') {
    knn.addExample(logits, 'true');
    console.log('true');
  } else if (key == 'F') {
    knn.addExample(logits, 'false');
    console.log('false');
  } else if (key == 'X') {
    save(knn, 'model.json');
    //knn.save('model.json');
  } else if (key == 'S') {
    start = true;
  } else if (key == 'L') {
    knn.load('./model.json', function () {
      console.log('knn loaded');
    });
  }
}


function modelReady() {
  console.log('model ready!');
  // Comment back in to load your own model!
  // knn.load('model.json', function() {
  //   console.log('knn loaded');
  // });
}

function goClassify() {
  const logits = features.infer(video);
  knn.classify(logits, function (error, result) {
    if (error) {
      console.error(error);
    } else {
      label = result.label;
      if (label == "true")
        background("#0fff4b");

      if (label == "false")
        background("#ff430f");

      goClassify();
    }
  });
}

// Temporary save code until ml5 version 0.2.2
const save = (knn, name) => {
  const dataset = knn.knnClassifier.getClassifierDataset();
  if (knn.mapStringToIndex.length > 0) {
    Object.keys(dataset).forEach(key => {
      if (knn.mapStringToIndex[key]) {
        dataset[key].label = knn.mapStringToIndex[key];
      }
    });
  }
  const tensors = Object.keys(dataset).map(key => {
    const t = dataset[key];
    if (t) {
      return t.dataSync();
    }
    return null;
  });
  let fileName = 'myKNN.json';
  if (name) {
    fileName = name.endsWith('.json') ? name : `${name}.json`;
  }
  saveFile(fileName, JSON.stringify({ dataset, tensors }));
};

const saveFile = (name, data) => {
  const downloadElt = document.createElement('a');
  const blob = new Blob([data], { type: 'octet/stream' });
  const url = URL.createObjectURL(blob);
  downloadElt.setAttribute('href', url);
  downloadElt.setAttribute('download', name);
  downloadElt.style.display = 'none';
  document.body.appendChild(downloadElt);
  downloadElt.click();
  document.body.removeChild(downloadElt);
  URL.revokeObjectURL(url);
};
