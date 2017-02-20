package com.example

import java.io.File
import java.net.MalformedURLException
import java.net.URL
import java.util.concurrent.*
import javax.imageio.IIOException
import javax.imageio.ImageIO

fun main(args: Array<String>) {
    val wnids = mapOf(
            "dog" to "n02084071",
            "cow" to "n01887787",
            "horse" to "n02374451",
            "cat" to "n02121808",
            "bear" to "n02131653",
            "fox" to "n02118333",
            "wolf" to "n02114100",
            "lion" to "n02129165",
            "tiger" to "n02129604"
    )
    wnids.forEach {
        var counter = 0
        val path = File(System.getProperty("user.dir"), "animals/${it.key}")
        if (!path.exists()) {
            path.mkdir()
        }
        val mutex = Semaphore(4)

        val url = URL("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=${it.value}")
        url.openStream().bufferedReader().forEachLine { s ->
            mutex.acquire()
            val downloadImage = Runnable {
                val future = Executors.newSingleThreadExecutor().submit {
                    try {
                        val imageurl = URL(s)
                        val image = ImageIO.read(imageurl)
                        if (image.width > 0) {
                            val format = s.split(".").last()
                            val file = File(path, "${it.key}-$counter.$format")
                            if (!file.exists()) {
                                file.createNewFile()
                                println("Downloading $s to ${file.name}\tQueue:${mutex.queueLength}")
                                ImageIO.write(image, format, file)
                            } else {
                                println("file already exist\tQueue:${mutex.queueLength}")
                            }
                            counter++
                        } else {
                            println("not downloading $s\tQueue:${mutex.queueLength}")
                        }
                    } catch (e: MalformedURLException) {
                        println("${e.message} $s")
                        mutex.release()
                    } catch (e: NullPointerException) {
                        println("${e.message} $s")
                        mutex.release()
                    } catch (e: IIOException) {
                        println("${e.message} $s")
                        mutex.release()
                    }
                }

                try {
                    future.get(10, TimeUnit.SECONDS)
                } catch (e: TimeoutException) {
                    println("timeout")
                    if (!future.isCancelled) {
                        future.cancel(true)
                    }
                    mutex.release()
                }
                mutex.release()
            }
            Thread(downloadImage).start()
            TimeUnit.MILLISECONDS.sleep(50)
        }
    }
}