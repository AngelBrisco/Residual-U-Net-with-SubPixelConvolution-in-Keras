# Residual-U-Net-with-SubPixelConvolution-in-Keras
In this model of a fully convolutional neural network I implement a subpixel convolution with ICNR initialization to mitigate the Checkboard artifact soo common in upsampling convolutions. The implementation it's done with a costume subclassed Keras layer with an also costume initializer so it`s compatible with Keras API.  
I've have used it in segmentation tasks as well as super-resolution ones with remarkable improbements over simple UpSampling2D convolutions.  
As drowback of my implementation I must acknoledge that its nor compatible with quantization or TfLite. Also because of the way that depth to space function works this layer can only be used to do 2d upscaling. It would be impractical to make a 3d one because we should augment the filters in a cubic way instead of a cuadratic one.

## Logaritmic scale
Aplaying the normal subpixel convolution you will notice that the number of filter needed increase in a cuadratic base. With this in mind you can calculate that upsampling by a factor of two will requiere Ci\*Co\*scale\*\*2, being Ci = Channels input and Co = Channels output. If you want to do a scale 2 upscaling you will need to do Ci\*Co\*2\*\*2 weights, but if you want to do a scale 8 upsampling you will need Ci\*Co\*8\*\*2. Thats meen that scale 2 needs Ci\*Co\*4 vs Ci\*Co\*64 of scale 8.  
As you would notice this is a significant increase in trainable parameters and in model size. But theres a trick that can be implemented because most of the time by the way that Conv layers work you have scales factors that can be expresed like 2\*\*x. So we can refactor the inner flow of the layer to do (Ci\*Co\*scale 2\*\*2)\*log2(desired scale) in a serial way and we would have the same results with a reduce footprint.  
For example while Ci\*Co\*8\*\*2 needs Ci\*Co\*64 weight to produce an 8 factor scaled image if we implement log2(8)=3 2 factor scaling we achive the same results with (Ci\*Co\*2\*\*2)\*3 = (Ci\*Co\*4)\*3 = Ci\*Co\*12. So this allows us to reduce the weights numbers from Ci\*Co\*64 to Ci\*Co\*12, 5.33 times less. This reduce model weights and training time considerable.

## Posible modifications
Instead of using a normal convolution it can be used a Separable convolution that will reduce the needed weights even more.
