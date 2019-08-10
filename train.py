import argparse
import torch
import os
import data_utils as du
import model_utils as mu

# Collect the input arguments
def process_arguments():
    ''' Collect the input arguments according to the syntax
        Return a parser with the arguments
    '''
    parser = argparse.ArgumentParser(description = 'Train new netword on a dataset and save the model')
    
    parser.add_argument('data_directory',
                       action='store',
                       default='flowers',
                       help='Input directory for training data')
    
    parser.add_argument('--save_dir',
                       action='store',
                       dest='save_directory', default='checkpoint_dir',
                       help='Directory where the checkpoint file is saved')
    
    parser.add_argument('--arch',
                       action='store',
                       dest='choosen_archi', default='vgg16',
                       help='Choosen pretrained architecture: vgg16 or densenet121')
    
    parser.add_argument('--learning_rate',
                       action='store',
                       dest='learning_rate', type=float, default=0.001,
                       help='Neural Network learning rate')
    
    parser.add_argument('--hidden_units',
                       action='store',
                       dest='hidden_units', type=int, default=512,
                       help='Number of hidden units')
    
    parser.add_argument('--epochs',
                       action='store',
                       dest='epochs', type=int, default=1,
                       help='Number of Epochs for the training')
    
    parser.add_argument('--gpu',
                       action='store_true',
                       default=False,
                       help='Use GPU. The default is CPU')
    
    return parser.parse_args()

# Get input arguments and train the specified network
def main():
    
    # Get the input arguments
    input_arguments = process_arguments()
    
    # Set the device to cuda if specified
    default_device = torch.device("cuda" if torch.cuda.is_available() and input_arguments.gpu else "cpu")
    
    # Set input_size for densenet121, by default
    input_size = 25088
    choosen_architecture = input_arguments.choosen_archi
    
    # Set input_size according to vgg13
    if (choosen_architecture == "densenet121"): input_size = 1024
    
    # Restrict archi to vgg16 or densenet121
    if (choosen_architecture != "vgg16") and (choosen_architecture != "densenet121"):
        print("Pretrained Choosen arhitecture is densenet121 or vgg16, using defaut: vgg16")
        choosen_architecture = "vgg16"
        
    # Load data
    train_data, test_data, valid_data, trainloader, testloader, validloader = du.loading_data(input_arguments.data_directory)
    
    # Set the choosen pretrained model
    model = mu.set_pretrained_model(choosen_architecture)
    
    # Set model classifier
    model = mu.set_model_classifier(model, input_arguments.hidden_units, input_size, output_size=102, dropout=0.05)
    
    # Train the model
    model, epochs, optimizer = mu.train_model(model, 
                                              trainloader, 
                                              input_arguments.epochs, 
                                              validloader, 
                                              input_arguments.learning_rate, 
                                              default_device)
    
    # Create a file path using the specified save_directory
    # to save the file as checkpoint.pth under that directory
    if not os.path.exists(input_arguments.save_directory):
        os.makedirs(input_arguments.save_directory)
    checkpoint_file_path = os.path.join(input_arguments.save_directory, "checkpoint.pth")
    
    # Store the trained model as checkpoint
    mu.create_checkpoint(model, 
                         input_arguments.choosen_archi,
                         train_data, 
                         epochs, 
                         optimizer, 
                         checkpoint_file_path,
                         input_size,
                         output_size=102)
    
    pass

if __name__ == '__main__':
    main()