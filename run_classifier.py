#!/usr/bin/env python3
import os
import argparse
from image_classifier import ImageClassifier

def main():
    parser = argparse.ArgumentParser(description="Run Image Classification Framework")
    
    # Required parameters
    parser.add_argument("--data_dir", required=True, 
                        help="Directory containing the image data organized in class folders")
    
    # Optional parameters
    parser.add_argument("--output_dir", default="results", 
                        help="Output directory for results and model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Initial learning rate")
    parser.add_argument("--model", default="resnet18", 
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="ResNet model variant to use")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Validation split ratio (0-1)")
    
    # Operation mode
    parser.add_argument("--mode", default="train", 
                        choices=["train", "evaluate", "predict"],
                        help="Operation mode: train (default), evaluate, or predict")
    parser.add_argument("--model_path", 
                        help="Path to pretrained model (for evaluate/predict modes)")
    parser.add_argument("--predict_image", 
                        help="Path to image for prediction (required for predict mode)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "predict" and not args.predict_image:
        parser.error("--predict_image is required when using --mode predict")
    
    if args.mode in ["evaluate", "predict"] and not args.model_path and not os.path.exists(os.path.join(args.output_dir, "final_model.pth")):
        parser.error("--model_path is required for evaluate/predict modes when no default model exists")
    
    # Create the classifier
    classifier = ImageClassifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model
    )
    
    # Execute based on mode
    if args.mode == "train":
        print("=== Starting Training Mode ===")
        classifier.load_data(val_split=args.val_split)
        classifier.create_model()
        classifier.train()
        classifier.evaluate()
        classifier.save_model()
        print("=== Training Complete ===")
        
    elif args.mode == "evaluate":
        print("=== Starting Evaluation Mode ===")
        classifier.load_data(val_split=args.val_split)
        if args.model_path:
            classifier.load_model(args.model_path)
        else:
            classifier.load_model()
        classifier.evaluate()
        print("=== Evaluation Complete ===")
        
    elif args.mode == "predict":
        print(f"=== Making Prediction for: {args.predict_image} ===")
        if args.model_path:
            classifier.load_model(args.model_path)
        else:
            classifier.load_model()
        result = classifier.predict(args.predict_image)
        print(f"Prediction: {result['class']} with {result['probability']:.2%} confidence")
        print("Top predictions:")
        for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {class_name}: {prob:.2%}")
        print(f"Visualization saved to {args.output_dir}/predictions/")

if __name__ == "__main__":
    main()