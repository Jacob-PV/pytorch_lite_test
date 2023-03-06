package com.example.myapplication;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private Module mModule;
    private List<String> mLocations = Arrays.asList("A","B","C","D","E","F","G","H","I","L","M","N","O","P");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
//            mModule = Module.load(assetFilePath(this, "changesToGraph.ptl"));
            mModule = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "ChangesToGraphMobile.ptl"));
        } catch (IOException e) {
            Log.d("errorlocation", "failed to load module");
            e.printStackTrace();
        }
        if (mModule == null) {
            Log.d("errorlocation", "module is null");
        }

        for (String location : mLocations) {
//            mModule.train();

//            float[] inputArray = {2.1440f, 2.0005f, 1.5840f};
//            FloatBuffer inputBuffer = Tensor.allocateFloatBuffer(inputArray.length).put(inputArray);
//            Tensor inputTensor = Tensor.fromBlob(inputBuffer, new long[]{1, 3});
//
//            float[] zerosArray = new float[1280];
//            Arrays.fill(zerosArray, 0f);
//            FloatBuffer zerosBuffer = Tensor.allocateFloatBuffer(zerosArray.length).put(zerosArray);
//            Tensor zerosTensor = Tensor.fromBlob(zerosBuffer, new long[]{10, 128});
//
////            IValue[] inputs = {IValue.from(inputTensor), IValue.from(zerosTensor), IValue.from(zerosTensor)};
////            Log.d("sampleIn", inputs.toString());
//
//
//            Tensor[] inputs = new Tensor[]{inputTensor, zerosTensor, zerosTensor};

            float[] inputArray = {2.1440f, 2.0005f, 15840f};
            FloatBuffer inputBuffer = Tensor.allocateFloatBuffer(inputArray.length).put(inputArray);
            Tensor inputTensor = Tensor.fromBlob(inputBuffer, new long[]{1, 3});

            float[] zerosArray = new float[1280];
            Arrays.fill(zerosArray, 0f);
            FloatBuffer zerosBuffer = Tensor.allocateFloatBuffer(zerosArray.length).put(zerosArray);
            Tensor zerosTensor = Tensor.fromBlob(zerosBuffer, new long[]{10, 128});

//            FloatBuffer fb = Tensor.allocateFloatBuffer(3);
//            fb.put(inputArray);
//            fb.put(zerosArray);
//            fb.put(zerosArray);
//            IValue iv = IValue.from(fb);

            List<Tensor> tuple = new ArrayList<>();
            tuple.add(zerosTensor);
            tuple.add(zerosTensor);

            IValue[] inputs = {IValue.from(inputTensor), IValue.from(zerosTensor), IValue.from(zerosTensor)};

            IValue outputTensor = mModule.forward(inputs);


//            float[] outputArray = outputTensor.getDataAsFloatArray();
            System.out.println(location + ": " + Arrays.toString(outputTensor.toTuple()[0].toTensor().getDataAsFloatArray()));


//            Tensor outputTensor = mModule.forward(IValue.listFrom(inputs)).toTensor();

//            float[] outputArray = outputTensor.getDataAsFloatArray();
//            System.out.println(location + ": " + Arrays.toString(outputArray));
        }

        System.out.println("Done");
    }

//    private String assetFilePath(Context context, String assetName) throws IOException {
//        return context.getFilesDir().getPath() + "/" + assetName;
//    }
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
