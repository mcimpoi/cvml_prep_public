
let mobilenet;
let model;
let isPredicting;
const dataset = new RPSDataset();

let rockSamples = 0;
let paperSamples = 0;
let scissorsSamples = 0;

const webcam = new Webcam(document.getElementById('wc'));

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        const classId = (await predictedClass.data())[0];
        var predictedText = "";
        switch (classId) {
            case 0:
                predictedText = "I see Rock";
                break;
            case 1: 
                predictedText = "I see Paper";
                break;  
            case 2:
                predictedText = "I see Scissors";
                break;
        }
        document.getElementById("prediction").innerText = predictedText;
        predictedClass.dispose();
        await tf.nextFrame();
    }
}

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(3);

    model = tf.sequential({
        layers: [
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            tf.layers.dense({units: 100, activation: 'relu'}),
            tf.layers.dense({units: 3, activation: 'softmax'})
        ]
    });

    const optimizer = tf.train.adam(0.0001);
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    let loss = 0;
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log('LOSS: ' + loss);
            }
        }
    });
}

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });
}

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
}

function handleButton(elem) {
    switch (elem.id) {
        case "0":
            ++rockSamples;
            document.getElementById("rocksamples").innerText = "Rock samples: " + rockSamples;
            break;
        case "1":
            ++paperSamples;
            document.getElementById("papersamples").innerText = "Paper samples: " + paperSamples;   
            break;
        case "2":
            ++scissorsSamples;
            document.getElementById("scissorssamples").innerText = "Scissors samples: " + scissorsSamples;
            break;
    }
    label = parseInt(elem.id);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);
}

function doTraining() {
    train();
}

init();