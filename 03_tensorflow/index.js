const model = tf.sequential();

const h1 = tf.layers.dense({
  units: 4,
  inputShape: [2],
  activation: "sigmoid",
});

model.add(h1);

const output = tf.layers.dense({
  units: 1,
  activation: "sigmoid",
});

model.add(output);

const opt = tf.train.sgd(0.1);

model.compile({
  optimizer: opt,
  loss: tf.losses.meanSquaredError,
});

const xs = tf.tensor2d([
  [0, 0],
  [0.5, 0.5],
  [1, 1],
]);

const ys = tf.tensor2d([[1], [0.5], [0]]);

train().then(() => {
  let output = model.predict(xs);

  console.log("Training Complete");
  output.print();
});

async function train() {
  for (let i = 0; i < 1000; i++) {
    const config = {
      shuffle: true,
    };
    const resp = await model.fit(xs, ys);
    console.log(resp.history.loss[0]);
  }
}
