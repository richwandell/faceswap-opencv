import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import org.bytedeco.javacpp.opencv_core.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

/**
 * Created by richwandell on 3/31/18.
 */
public class Main {

    private static float[][] landmarks1 = new float[][]{
            {29f, 60f},
            {30f, 67f},
            {31f, 75f},
            {32f, 82f},
            {35f, 89f},
            {38f, 96f},
            {44f, 101f},
            {51f, 105f},
            {59f, 106f},
            {67f, 105f},
            {73f, 101f},
            {78f, 95f},
            {82f, 88f},
            {84f, 81f},
            {85f, 73f},
            {86f, 65f},
            {86f, 57f},
            {33f, 49f},
            {35f, 43f},
            {40f, 40f},
            {45f, 39f},
            {51f, 40f},
            {61f, 40f},
            {67f, 38f},
            {72f, 38f},
            {77f, 41f},
            {80f, 46f},
            {57f, 48f},
            {57f, 52f},
            {57f, 56f},
            {58f, 61f},
            {52f, 68f},
            {55f, 68f},
            {58f, 69f},
            {61f, 68f},
            {63f, 67f},
            {39f, 53f},
            {41f, 51f},
            {45f, 50f},
            {49f, 52f},
            {46f, 53f},
            {42f, 54f},
            {64f, 52f},
            {68f, 49f},
            {72f, 50f},
            {75f, 52f},
            {72f, 52f},
            {68f, 52f},
            {45f, 82f},
            {49f, 78f},
            {54f, 76f},
            {58f, 76f},
            {62f, 75f},
            {67f, 77f},
            {72f, 81f},
            {68f, 87f},
            {63f, 90f},
            {58f, 90f},
            {54f, 90f},
            {49f, 88f},
            {47f, 82f},
            {54f, 79f},
            {58f, 79f},
            {62f, 79f},
            {70f, 81f},
            {62f, 85f},
            {58f, 85f},
            {54f, 85f}
    };

    private static float[][] landmarks2 = new float[][]{
        {144f, 159f},
        {147f, 180f},
        {149f, 201f},
        {153f, 222f},
        {161f, 241f},
        {176f, 255f},
        {194f, 268f},
        {214f, 278f},
        {235f, 279f},
        {254f, 274f},
        {269f, 261f},
        {282f, 248f},
        {292f, 232f},
        {296f, 213f},
        {296f, 192f},
        {296f, 172f},
        {297f, 152f},
        {162f, 147f},
        {170f, 135f},
        {184f, 127f},
        {201f, 125f},
        {215f, 130f},
        {239f, 129f},
        {252f, 123f},
        {268f, 124f},
        {281f, 131f},
        {288f, 143f},
        {229f, 144f},
        {229f, 156f},
        {230f, 168f},
        {231f, 181f},
        {217f, 194f},
        {224f, 195f},
        {232f, 197f},
        {239f, 194f},
        {245f, 192f},
        {183f, 151f},
        {190f, 147f},
        {199f, 146f},
        {207f, 150f},
        {199f, 152f},
        {191f, 153f},
        {246f, 148f},
        {254f, 143f},
        {263f, 143f},
        {270f, 147f},
        {263f, 149f},
        {255f, 149f},
        {205f, 222f},
        {216f, 219f},
        {225f, 215f},
        {233f, 216f},
        {241f, 214f},
        {250f, 215f},
        {260f, 216f},
        {252f, 223f},
        {244f, 227f},
        {236f, 228f},
        {228f, 229f},
        {217f, 227f},
        {210f, 222f},
        {226f, 221f},
        {234f, 220f},
        {241f, 219f},
        {256f, 217f},
        {242f, 219f},
        {235f, 221f},
        {227f, 221f}
    };



    private static INDArray fixColors(INDArray badMatrix) {
        INDArray a = badMatrix.getRow(0);
        INDArray r = a.getRow(0);
        INDArray g = a.getRow(1);
        INDArray b = a.getRow(2);

        return a;
    }

    private static void writeIndArrayToDisk(INDArray imagePixels) {

        INDArray a = imagePixels.getRow(0);
        INDArray b = a.getRow(0);
        INDArray g = a.getRow(1);
        INDArray r = a.getRow(2);

        int[] shape = r.shape();
        int height = shape[0];
        int width = shape[1];

        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for( int i = 0; i < width * height; i++ ){
            int blue = (int)b.getDouble(i);
            int green = (int)g.getDouble(i);
            int red = (int)r.getDouble(i);

            WritableRaster raster = bi.getRaster();
            raster.setSample(
                    i % width,
                    i / width,
                    0,
                    red
            );
            raster.setSample(
                    i % width,
                    i / width,
                    1,
                    green
            );
            raster.setSample(
                    i % width,
                    i / width,
                    2,
                    blue
            );
        }

        try {
            ImageIO.write(bi, "jpg", new File("outfile.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Loader.load(opencv_java.class);
    }

    public static void main(String[] args) {

        try {
            Mat im1 = Imgcodecs.imread("./resources/my-face.jpg");
            Mat im2 = Imgcodecs.imread("./resources/brad-face.jpg");

            FaceSwapper f = new FaceSwapper(im1, im2, landmarks1, landmarks2);



        }catch(Exception e){
            System.out.println(e.toString());
        }
    }


}
