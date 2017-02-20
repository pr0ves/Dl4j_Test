package com.example

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File

fun main(args: Array<String>) {
    val net = ModelSerializer.restoreMultiLayerNetwork(File(System.getProperty("user.dir"), "net.zip"))
    val mnistTest = MnistDataSetIterator(batchSize, 500)
    while (mnistTest.hasNext()) {
        val ds = mnistTest.next()
        for (i in 0..ds.featureMatrix.rows()) {
            val image = ds.featureMatrix.getRow(i)
            val draw = DrawReconstruction(image.mul(255))
            draw.draw()
            printProbabilities(net.output(image))
            System.`in`.read()
            draw.close()
        }
    }
}

fun printProbabilities(output: INDArray){
    if (output.rows()!=1&&output.columns()!=10){
        println("wrong indarray")
        return
    }
    for (i in 0..output.columns()-1) {
        println("$i: ${output.mul(100).getScalar(i)}")
    }
}