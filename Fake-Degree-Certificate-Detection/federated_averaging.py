import tensorflow as tf
import os
import sys

def main():
    model1_path = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\fake_certificate_model.h5"
    model2_path = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\model2.h5"
    global_model_path = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\global_model.h5"

    print(f"Loading Model 1 from: {model1_path}")
    if not os.path.exists(model1_path):
        print(f"Error: Model 1 not found at {model1_path}")
        sys.exit(1)
    model1 = tf.keras.models.load_model(model1_path)
    
    print(f"Loading Model 2 from: {model2_path}")
    if not os.path.exists(model2_path):
        print(f"Error: Model 2 not found at {model2_path}")
        sys.exit(1)
    model2 = tf.keras.models.load_model(model2_path)

    print("Extracting weights from both models...")
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    print("Verifying architectures match based on trainable and non-trainable parameter tensors...")
    if len(weights1) != len(weights2):
        print(f"Mismatch: Model 1 has {len(weights1)} weight tensors, but Model 2 has {len(weights2)}.")
        sys.exit(1)
        
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        if w1.shape != w2.shape:
            print(f"Mismatch: Weight tensor {i} shape differs ({w1.shape} vs {w2.shape}).")
            sys.exit(1)

    print("Architectures validated. Performing layer-wise Federated Averaging...")
    global_weights = [(w1 + w2) / 2.0 for w1, w2 in zip(weights1, weights2)]
    
    print("Creating global model using clone_model()...")
    # Clone Model 1 because it has the exact preprocessing layers built-in.
    global_model = tf.keras.models.clone_model(model1)
    
    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    print("Assigning averaged weights to the new global model...")
    global_model.set_weights(global_weights)
    
    print(f"Saving final aggregated model to '{global_model_path}'...")
    global_model.save(global_model_path)
    print("✅ Federated Averaging completed successfully!")

if __name__ == "__main__":
    main()
