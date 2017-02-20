package com.example

import org.apache.commons.io.FileUtils
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.FlipImageTransform
import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.WarpImageTransform
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.inputs.InputType
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator



val imagePath = File(System.getProperty("user.dir"), "animals/")
val height = 100
val width = 100
val numExamples = 80
val numLabels = 4
val batchSize = numExamples / numLabels
val seed: Long = 42
val rng = Random(seed)
val splitTrainTest = 0.8
val iterations = 5
val epochs = 50
val nCores = 8
var channels = 3

fun main(args: Array<String>) {
    val log = LoggerFactory.getLogger("Main")!!
    log.info("Loading data...")
    getExamples()
    log.info("Done")

    val labelMaker = ParentPathLabelGenerator()
    val fileSplit = FileSplit(File(imagePath, "test/"), NativeImageLoader.ALLOWED_FORMATS, rng)
    val pathFilter = BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize)

    val inputSplit = fileSplit.sample(pathFilter, numExamples * (1 + splitTrainTest), numExamples * (1 - splitTrainTest))
    val trainData = inputSplit[0]
    val testData = inputSplit[1]

    val transforms = arrayOf<ImageTransform>(FlipImageTransform(rng), WarpImageTransform(rng, seed.toFloat()), FlipImageTransform(Random(123)))

    val scaler: DataNormalization = ImagePreProcessingScaler(0.0, 1.0)

    val net: MultiLayerNetwork = alexnetModel()

    net.init()
    net.setListeners(ScoreIterationListener(1))

    val recordReader = ImageRecordReader(height, width, channels, labelMaker)
    var dataIter: DataSetIterator
    var trainIter: MultipleEpochsIterator

    log.info("Train model....")
    recordReader.initialize(trainData, null)
    dataIter = RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    scaler.fit(dataIter)
    dataIter.preProcessor = scaler
    trainIter = MultipleEpochsIterator(epochs, dataIter, nCores)
    net.fit(trainIter)

    transforms.forEach {
        log.info("Training on transformation: ${it.javaClass}\n\n")
        recordReader.initialize(trainData, it)
        dataIter = RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
        scaler.fit(dataIter)
        dataIter.preProcessor = scaler
        trainIter = MultipleEpochsIterator(epochs, dataIter, nCores)
        net.fit(trainIter)
    }

    log.info("Evaluate model....")
    recordReader.initialize(testData)
    dataIter = RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    scaler.fit(dataIter)
    dataIter.preProcessor = scaler
    val eval = net.evaluate(dataIter)
    log.info(eval.stats(true))


    dataIter.reset()
    val testDataSet = dataIter.next()
    val expectedResult = testDataSet.getLabelName(0)
    val predict = net.predict(testDataSet)
    val modelResult = predict[0]
    log.info("\nFor a single example that is labeled $expectedResult the model predicted $modelResult\n\n")

    FileUtils.deleteDirectory(File(imagePath, "test/"))

    log.info("Save model....")
    ModelSerializer.writeModel(net, File(System.getProperty("user.dir", "net.zip")), true)

}

fun getExamples() {
    val dir = File(imagePath, "test/")
    dir.mkdir()
    imagePath.listFiles { file -> file.isDirectory }.forEach {
        if (it.name != "test") {
            val tmpDir = File(dir, it.name)
            tmpDir.mkdir()
            var i = 0
            val nFiles = it.listFiles().size

            while (i < (numExamples / numLabels)) {
                val rand = Random().nextInt(nFiles)
                val tmpFile = File(tmpDir, it.listFiles()[rand].name)
                if (!tmpFile.exists()) {
                    tmpFile.createNewFile()
                    Files.copy(it.listFiles()[rand].toPath(), tmpFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
                    i++
                }
            }
        }
    }
}

fun convInit(name: String, `in`: Int, out: Int, kernel: IntArray, stride: IntArray, pad: IntArray, bias: Double): ConvolutionLayer {
    return ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(`in`).nOut(out).biasInit(bias).build()
}

fun conv3x3(name: String, out: Int, bias: Double): ConvolutionLayer {
    return ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1), intArrayOf(1, 1)).name(name).nOut(out).biasInit(bias).build()
}

fun conv5x5(name: String, out: Int, stride: IntArray, pad: IntArray, bias: Double): ConvolutionLayer {
    return ConvolutionLayer.Builder(intArrayOf(5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build()
}

fun maxPool(name: String, kernel: IntArray): SubsamplingLayer {
    return SubsamplingLayer.Builder(kernel, intArrayOf(2, 2)).name(name).build()
}

fun fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution): DenseLayer {
    return DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build()
}

fun alexnetModel(): MultiLayerNetwork {
    /**
     * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
     * and the imagenetExample code referenced.
     * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
     */

    val nonZeroBias = 1.0
    val dropOut = 0.5

    val conf = NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .biasLearningRate(1e-2 * 2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000.0)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(true)
            .list()
            .layer(0, convInit("cnn1", channels, 96, intArrayOf(11, 11), intArrayOf(4, 4), intArrayOf(3, 3), 0.0))
            .layer(1, LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, maxPool("maxpool1", intArrayOf(3, 3)))
            .layer(3, conv5x5("cnn2", 256, intArrayOf(1, 1), intArrayOf(2, 2), nonZeroBias))
            .layer(4, LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, maxPool("maxpool2", intArrayOf(3, 3)))
            .layer(6, conv3x3("cnn3", 384, 0.0))
            .layer(7, conv3x3("cnn4", 384, nonZeroBias))
            .layer(8, conv3x3("cnn5", 256, nonZeroBias))
            .layer(9, maxPool("maxpool3", intArrayOf(3, 3)))
            .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, GaussianDistribution(0.0, 0.005)))
            .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, GaussianDistribution(0.0, 0.005)))
            .layer(12, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .name("output")
                    .nOut(numLabels)
                    .activation(Activation.SOFTMAX)
                    .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build()

    return MultiLayerNetwork(conf)

}

fun img2DataSet(file: File): DataSet? {
    val recordReader = ImageRecordReader(height, width, 3)
    recordReader.initialize(FileSplit(File(file.parent)))
    val dataiter = RecordReaderDataSetIterator(recordReader, 1)
    if (dataiter.hasNext()) {
        return dataiter.next()
    }
    return null
}