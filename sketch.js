var trainButton;
var saveButton;
var drawButton;
var guessButton;

const EPOCHS = 10;

let train_data;
let train_labels;
let test_data;
let test_labels;

let train_data_bin;
let train_labels_bin;
let test_data_bin;
let test_labels_bin;

let isTraining = false;
let isDrawing = false;

let canvas;

let model = tf.sequential();

async function preload() {
	train_data_bin   = loadBytes('data/full_binary_butterfly.bin');
	train_labels_bin = loadBytes("data/train_labels.bin");
	test_data_bin    = loadBytes("data/test_data.bin");
	test_labels_bin  = loadBytes("data/test_labels.bin");
}

function prepareDataset(){
	console.log('preparing training data...');

	console.log(train_data_bin.bytes);
	console.log(train_data_bin.bytes.length % 784);
	//console.log(max(train_data_bin.bytes[0]),min(train_data_bin.bytes));

	let data = [];
	for (var i = 0; i < 784 * 600; i++) {
		data.push((test_data_bin.bytes[i] & 0xff) / 255);
	}
	train_data = tf.tensor(data,[600 ,28,28,1]);

	console.log('preparing testing data...');

	data = [];
	for (var i = 0; i < test_data_bin.bytes.length; i++) {
		data.push((test_data_bin.bytes[i] & 0xff) / 255);	
	}
	test_data = tf.tensor(data,[10000 ,28,28,1]);

	console.log('preparing training label...');
		
	train_labels = tf.oneHot(tf.tensor1d(train_labels_bin.bytes.slice(0,600), 'int32'), 10);
	
	console.log('preparing testing label...');

	test_labels = tf.oneHot(tf.tensor1d(test_labels_bin.bytes, 'int32'), 10);

	print('Succesfully prepared all dataset!');

	// //==
	// train = []
	// for (var i = 0; i < 784 * 1000; i++) {
	// 	train.push(0);
	// }
	// for (var i = 0; i < 784 * 1000; i++) {
	// 	train.push(1);
	// }
	// train_data = tf.tensor(train,[2000,28,28]);

	// train = [];
	// for (var i = 0; i < 1000; i++) {
	// 	train.push(0);
	// }
	// for (var i = 0; i < 1000; i++) {
	// 	train.push(1);
	// }
	// train_labels = tf.oneHot(tf.tensor1d(train, 'int32'), 10);
	// //==

	// console.log(train_data.arraySync());
	// console.log(train_labels.arraySync());
	// console.log(test_data.arraySync());
	// console.log(test_labels.arraySync());
}

function setup() {
	createCanvas(600, 600);
	background(51);
	textAlign(CENTER, CENTER);
	fill(255);
	text('TRAINING . . .',width / 2, height / 2);

	console.log('started');
	canvas = createGraphics(28,28);
	prepareDataset();
	createModel();

	
	// trainModel();
	

	trainButton = createButton(`TRAIN ${EPOCHS} EPOCHS`);
	trainButton.mousePressed(() => trainModel());

	saveButton = createButton('SAVE MODEL');
	saveButton.mousePressed(() => {
		model.save('downloads://Mnist-model');
	});

	guessButton = createButton('GUESS');
	guessButton.mousePressed(() => predictScreen());

	drawButton = createButton('DRAW MODE');
	drawButton.mousePressed(() => {
		isDrawing=!isDrawing
		canvas.background(255);
	});
	// frameRate(1);
}

function draw() {
	if(!isTraining && !isDrawing){
		canvas.loadPixels();

		let n = floor(random(600));

		let data = train_data.arraySync()[n];
		for (var i=0; i<28; i++) {
			for (var j=0; j<28; j++) {
				let features  = 255 * (1 - data[i][j][0]);

				canvas.pixels[(j + 28 * i) * 4    ]=features;
				canvas.pixels[(j + 28 * i) * 4 + 1]=features;
				canvas.pixels[(j + 28 * i) * 4 + 2]=features;
				canvas.pixels[(j + 28 * i) * 4 + 3]=255;
			}
		}
		
		canvas.updatePixels();
		console.log('label:',indexOfMax(train_labels.arraySync()[n]));
		predictScreen();
		console.log(tf.memory().numTensors);
	}
	image(canvas,0,0,width,height);
}

function mouseDragged(){
	if(!isTraining && isDrawing){
		canvas.stroke(0);
		canvas.strokeWeight(1);
		canvas.line(pmouseX * (28 / width),pmouseY * (28 / height),mouseX * (28 / width),mouseY  * (28 / height));
	}
}

function keyPressed(){
	canvas.background(255);
}

function predictScreen(){
	// if(isDrawing){
	// 	loadPixels();
	// 	let data = [];
	// 	for (var i=0; i<28; i++) {
	// 		for (var j=0; j<28; j++) {
	// 			data.push(1 - (pixels[(j + 28 * i) * 4]) / 255);
	// 		}
	// 	}
	// 	console.log(data);
	// 	tf.tidy(() => {
	// 		console.log('guess: ',indexOfMax(model.predict(tf.tensor([data],[1,28,28,1])).dataSync()));
	// 	});
	// }
	// else{
	canvas.loadPixels();
	let data = [];
	for (var i=0; i<28; i++) {
		for (var j=0; j<28; j++) {
			data.push(1 - (canvas.pixels[(j + 28 * i) * 4]) / 255);
		}
	}
	tf.tidy(() => {
		console.log('guess: ',indexOfMax(model.predict(tf.tensor([data],[1,28,28,1])).dataSync()));
	});
	// }
}

async function trainModel(){
	isTraining = true;

	let response = await model.fit(train_data, train_labels,{
		batchSize: 20,
		epochs: EPOCHS,
		shuffle: true,
		validationData:[test_data,test_labels]
	});


	console.log('loss:',response.history);
	console.log(response);	

	isTraining = false;
}

async function createModel(){

	// model.add(tf.layers.conv2d({
	// 	inputShape: [28,28,1],
	// 	filters: 128,
	// 	kernelSize: 3,
	// 	activation: 'relu'
	// }));

	// model.add(tf.layers.maxPooling2d({
	// 	poolSize: 2
	// }));

	// model.add(tf.layers.conv2d({
	// 	filters: 128,
	// 	kernelSize: 3,
	// 	activation: 'relu'
	// }));

	// model.add(tf.layers.maxPooling2d({
	// 	poolSize: 2
	// }));

	// model.add(tf.layers.conv2d({
	// 	filters: 64,
	// 	kernelSize: 3,
	// 	activation: 'relu'
	// }));

	// model.add(tf.layers.maxPooling2d({
	// 	poolSize: 2
	// }));

	// model.add(tf.layers.flatten());;

	// model.add(tf.layers.dense({
	// 	units: 10,
	// 	activation: 'softmax'
	// }));

	isTraining = true;
	model = await tf.loadLayersModel('models/Mnist-model.json');

	await model.compile({
		optimizer: tf.train.adam(0.01),
		loss: tf.losses.meanSquaredError,
		metrics:tf.metrics.sparseCategoricalAccuracy 
	});

	isTraining = false;

	// result = model.evaluate(train_data, train_labels,{batchSize: 20});
	// result.print();
	// result = model.evaluate(test_data, test_labels,{batchSize: 20});
	// result.print();
}


// function oneHot(label){
// 	let oneHot = [];
// 	for(var i = 0; i < 10; i++){
// 		if(label == i)
// 			oneHot.push(1);
// 		else
// 			oneHot.push(0);
// 	}

// 	return oneHot;
// }

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

// console.log('preparing training data...');

// 	let data = [];
// 	for (var i = 0; i < train_data_bin.bytes.length / 1000; i++) {
// 		data.push((255 - train_data_bin.bytes[i] & 0xff) / 255);
// 	}
// 	train_data = tf.tensor(data,[60000 / 1000,784]);

// 	console.log('preparing testing data...');

// 	data = [];
// 	for (var i = 0; i < test_data_bin.bytes.length / 1000; i++) {
// 		data.push((255 - test_data_bin.bytes[i] & 0xff) / 255);	
// 	}
// 	test_data = tf.tensor(data,[10000 / 1000,784]);

// 	console.log('preparing training label...');
		
// 	train_labels = tf.oneHot(tf.tensor1d(train_labels_bin.bytes.slice(0,60), 'int32'), 10);
	
// 	console.log('preparing testing label...');

// 	test_labels = tf.oneHot(tf.tensor1d(test_labels_bin.bytes.slice(0,10), 'int32'), 10);

// 	print('Succesfully prepared all dataset!');


	// model.add(tf.layers.flatten({inputShape: [28,28]}));

	// model.add(tf.layers.dense({
	// 	units: 128,
	// 	activation: 'sigmoid',
	// }));

	// // model.add(tf.layers.dropout({
	// // 	rate: 0.2
	// // }));

	// // model.add(tf.layers.dropout({
	// // 	rate: 0.1
	// // }));

	// model.add(tf.layers.dense({
	// 	units: 128,
	// 	activation: 'sigmoid',
	// }));

	// // model.add(tf.layers.dropout({
	// // 	rate: 0.1
	// // }));

	// model.add(tf.layers.dense({
	// 	units: 10,
	// 	activation: 'softmax',
	// }));



	// //model = await tf.loadLayersModel('models/my-model.json');

	// model.compile({
	// 	optimizer: tf.train.adam(0.2),
	// 	loss: tf.losses.meanSquaredError,
	// });