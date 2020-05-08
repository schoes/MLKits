// require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('./load-csv');

const knn = (features, labels, predictionPoint, k) => {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPredictionPoint = predictionPoint
        .sub(mean)
        .div(variance.pow(0.5));
    return (
        features
            .sub(mean)
            .div(variance.pow(0.5))
            .sub(scaledPredictionPoint)
            .pow(2)
            .sum(1)
            .pow(0.5)
            .expandDims(1)
            .concat(labels, 1)
            .unstack()
            .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
            .slice(0, k)
            .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
    );
};

let { features, labels, testFeatures, testLabels } = loadCsv(
    'kc_house_data.csv',
    {
        shuffle: true,
        splitTest: 10,
        dataColumns: ['lat', 'long', 'sqft_lot','sqft_living'],
        labelColumns: ['price'],
    }
);

testFeatures.forEach((testPoint, i) => {
    const result = knn(
        tf.tensor(features),
        tf.tensor(labels),
        tf.tensor(testPoint),
        10
    );
    const errorPercentage =
        ((testLabels[i][0] - result) / testLabels[i][0]) * 100;
    console.log('Error: ', errorPercentage);
});
