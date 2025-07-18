# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

 #from sklearn.model_selection import train_test_split
 #from sklearn.model_selection import StratifiedShuffleSplit
 #from sklearn.metrics import ConfusionMatrixDisplay
 #from sklearn import metrics

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from temperature_ros.LSTMNet import LSTMNet


# Dataset definition:
class MaterialDataset(Dataset):    
    """
    PyTorch Dataset for material classification from temperature time series.

    This class loads CSV files containing timestamped temperature data for multiple experiments,
    assigns labels, performs optional data augmentation, splits data into train/test sets,
    and provides access to individual time series sequences in a format suitable for training 
    neural networks.

    Parameters:
        path_to_csv (str): Path to the directory containing .csv files.
        which (int): Determines which dataset view to use:
                     0 - Full dataset
                     1 - Augmented dataset
                     2 - Training dataset
                     3 - Test dataset

    Attributes:
        df (DataFrame): Raw combined dataset from all .csv files.
        augmented_df (DataFrame): Augmented version of the dataset with Gaussian noise.
        train_df (DataFrame): Training split of the dataset.
        test_df (DataFrame): Testing split of the dataset.
        class_names (list): List of label names inferred from the 'notes' column.

    Methods:
        merge(df1, df2): Merges two dataframes with experiment index offset.
        split(augmented=True): Stratified split into train/test.
        data_augmentation(times=10, noise_std=0.001): Adds noise and generates subsamples.
        resample(timeseries, timestamp, frequency=10): Resamples the data at fixed frequency.
        plot(): Visualizes sequences colored by class.
    """
    def __init__(self, path_to_csv:str, which=1):
        """
        Initializes the dataset by loading CSV files, assigning labels, and processing timestamps.

        Parameters:
            path_to_csv (str): Path to the folder containing CSV annotation files.
            which (int): Controls which dataset to load:
                         0 - original dataset,
                         1 - augmented,
                         2 - training set,
                         3 - test set.
        """
        annotations_files = [f for f in os.listdir(path_to_csv) if f.endswith('.csv')]
        annotations_files.sort()
        # Check if the dataset is for training or testing # 0 full dataset, 1 augmented, 2 train, 3 test
        self.which = which
        # Initialize DataFrames
        self.df = pd.DataFrame(columns=["Timeseries ID","Experiment","Timestamp","temperature [C]","delta_t [C/s]","Label","notes"])
        self.augmented_df = pd.DataFrame(columns=["Timeseries ID","Experiment","Timestamp","temperature [C]","delta_t [C/s]","Label","notes"])
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()

        self.class_names = []
        
        # Load all annotations files
        for file_idx, file in enumerate(annotations_files):
            # iterate over each tissue file
            curr_df = pd.read_csv(os.path.join(path_to_csv, file))

            self.class_names.append(os.path.splitext(os.path.basename(file))[0])
            #self.class_names.append(curr_df['notes'].dropna().unique().astype(str))         # Get dataset labelized name

            # Assign numeric labels to the current DataFrame
            curr_df["Label"] = [file_idx] * len(curr_df)
            self.df = pd.concat([self.df, curr_df], ignore_index=True)

        ## Load all annotations files
        #for file_idx, file in enumerate(annotations_files):
        #    # iterate over each tissue file
        #    curr_df = pd.read_csv(os.path.join(path_to_csv, file))
        #    
        #    self.class_names.append(os.path.splitext(os.path.basename(file))[0])
        #    #self.class_names.append(curr_df['notes'].dropna().unique().astype(str))         # Get dataset labelized name
        #
        #    # Assign numeric labels to the current DataFrame
        #    curr_df["Label"] = [file_idx] * len(curr_df)
        #    self.df = pd.concat([self.df, curr_df], ignore_index=True)
        
        # Assign timeseries ID
        group_cols = ['Experiment', 'Label']
        df_groups = self.df.groupby(group_cols)
        for timeseries_id, (group_key, group_indices) in enumerate(df_groups.groups.items()):
            # Assign the Timeseries ID back to df
            self.df.loc[group_indices, 'Timeseries ID'] = timeseries_id
            # Shift 'Timestamp' to t0=0 at each experiment and convert into seconds
            self.df.loc[group_indices, 'Timestamp'] = 1e-9*(self.df.loc[group_indices, 'Timestamp'] - self.df.loc[group_indices, 'Timestamp'].min())
    
    def merge(self, df1:pd.DataFrame, df2:pd.DataFrame):
        """
        Merges two DataFrames and shifts the 'Experiment' IDs in df2 to avoid collisions.

        Saves the result to a CSV file on disk.

        Parameters:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame to be merged with shifted experiment IDs.
        """
        experiment_shift = df1['Experiment'].max()
        df2['Experiment'] = df2['Experiment'] + experiment_shift 
        merged_df = pd.concat([df1, df2], ignore_index=True)
        merged_df.to_csv('/home/leo/Desktop/temperature_ws/src/temperature_ros/config/net/data/merged.csv', index=False)

    #def split(self, augmented = True, test_size=0.2, random_state=42):
    #    """
    #    Splits the dataset into training and test sets.
    #
    #    Parameters:
    #        augmented (bool): Whether to use the augmented dataset for splitting.
    #        test_size (float): Proportion of the dataset to use for testing.
    #        random_state (int): Random seed for reproducibility.
    #    """
    #    df = self.augmented_df if augmented else self.df
    #
    #    train_df, test_df = train_test_split(df, test_size, random_state, stratify=df[['Timeseries ID','Label']])
    #    self.train_df = train_df.sort_values(by=['Timeseries ID','Timestamp']).reset_index(drop=True)
    #    self.test_df = test_df.sort_values(by=['Timeseries ID','Timestamp']).reset_index(drop=True)
    
    def data_augmentation(self, times=10, noise_std=0.001):
        """
        Augments the dataset by duplicating each time series and adding Gaussian noise.

        Parameters:
            times (int): Number of augmented copies per original time series.
            noise_std (float): Standard deviation of the Gaussian noise.
        """
        self.augmented_df = pd.DataFrame(columns=["Timeseries ID","Experiment","Timestamp","temperature [C]","delta_t [C/s]","Label","notes"])
        # Create new DataFrame for subsamples
        tmp_df = pd.DataFrame(columns=["subsample","Timeseries ID","Experiment","Timestamp","temperature [C]","delta_t [C/s]","Label","notes"])
        tmp_df = pd.concat([tmp_df, self.df], ignore_index=True)

        seq = np.arange(1, times+1)
        repeated_seq = np.tile(seq, int(np.ceil(len(tmp_df) / len(seq))))[:len(tmp_df)]
        tmp_df['subsample'] = repeated_seq

        # Group by timeseries and subsamples
        tmp2_df = pd.DataFrame(columns=["subsample","Timeseries ID","Experiment","Timestamp","temperature [C]","delta_t [C/s]","Label","notes"])
        group_cols = ['Timeseries ID', 'subsample']
        df_groups = tmp_df.groupby(group_cols)
        for timeseries_id, (group_key, group_indices) in enumerate(df_groups.groups.items()):
            # Assign the Timeseries ID back to df
            group_df = self.df.loc[group_indices]
            group_df['Timeseries ID'] = timeseries_id
            tmp2_df = pd.concat([tmp2_df, group_df], ignore_index=True)

        # Perturbate each timeserie with gaussian noise        
        tmp2_df['temperature [C]'] = tmp2_df['temperature [C]'] + np.random.normal(0, noise_std, size=len(tmp2_df['temperature [C]']))

        # Store the augmented DataFrame
        self.augmented_df = tmp2_df[["Timeseries ID","Experiment","Timestamp","temperature [C]","delta_t [C/s]","Label","notes"]]
        self.augmented_df = self.augmented_df.sort_values(by=['Timeseries ID','Timestamp']).reset_index(drop=True)

    def resample(self, timeseries:np.array, timestamp:np.array, frequency=10):    
        """
        Resamples a temperature time series to a uniform frequency.

        Parameters:
            timeseries (np.array): Array of temperature values.
            timestamp (np.array): Corresponding timestamps (in seconds).
            frequency (int): Desired sampling frequency in Hz.

        Returns:
            tuple: (resampled_timeseries, resampled_timestamp)
        """
        duration = (timestamp[-1] - timestamp[0])
        # Resample the timeseries to the specified frequency
        resampled_timestamp = np.linspace(timestamp[0], timestamp[-1], int(np.ceil(frequency * duration)))
        resampled_timeseries = np.interp(resampled_timestamp, timestamp, timeseries)
        return (resampled_timeseries, resampled_timestamp)
    

    def plot(self): 
        """
        Plots all the time series in the dataset with different colors for each class.
        Only one sample per class is shown in the legend.
        """   
        plotted_labels = []
        sizes = [0, 0, 0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        plt.figure()
        for i in range(len(self)):
            item = self[i]
            seq = item[0].numpy()
            lbl = item[1].item()
            seq = seq[:,0].reshape((len(seq[:,0]),))
            # Get the label name from the categorical codes 
            label_name = self.class_names[lbl]
            #increment dataset counter
            sizes[lbl] += 1
            # Plot the sequence with the label name
            if label_name not in plotted_labels:
                plt.plot(seq, color=colors[lbl], label=label_name)
                plotted_labels.append(label_name)
            else:
                plt.plot(seq, color=colors[lbl], label='_nolegend_')
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (C)')
        plt.title('Training Sequences: Temperature over Time')
        plt.legend(title="Class", loc="upper right")
        plt.show()
        
        # Create the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=self.class_names, autopct='%1.1f%%', startangle=90)
        plt.title('Data')
        plt.axis('equal')  # Equal aspect ratio ensures the pie is a circle
        plt.show()

    def __len__(self):
        """
        Returns the number of unique time series (based on Timeseries ID) in the selected subset.

        Returns:
            int: Number of samples.
        """
        if self.which == 1:
            df = self.augmented_df
        elif self.which == 2:
            df = self.df_train
        elif self.which == 3:
            df = self.df_test
        else:
            df = self.df
            
        return df['Timeseries ID'].values.max() + 1  # Return the maximum Timeseries ID + 1

    def __getitem__(self, idx):
        """
        Retrieves a resampled temperature sequence and its label for a given index.

        Parameters:
            idx (int): Index of the time series.

        Returns:
            tuple: (features as torch tensor [T,1], label as torch tensor)
        """
        if self.which == 1:
            df = self.augmented_df
        elif self.which == 2:
            df = self.df_train
        elif self.which == 3:
            df = self.df_test
        else:
            df = self.df

        raw_seq = np.array(df.loc[df['Timeseries ID'] == idx, 'temperature [C]'].values).astype(float)
        draw_seq = np.array(df.loc[df['Timeseries ID'] == idx, 'delta_t [C/s]'].values).astype(float)
        #raw_seq = np.array(df.loc[df['Timeseries ID'] == idx, 'delta_t [C/s]'].values).astype(float)
        raw_times = np.array(df.loc[df['Timeseries ID'] == idx, 'Timestamp'].values).astype(float)
        # Convert to interpolated features
        (seq,_) = self.resample(raw_seq, raw_times)
        (dseq,_) = self.resample(draw_seq, raw_times)

        # Tensorize the sequence
        seq = seq.reshape((len(seq),1))
        dseq = dseq.reshape((len(dseq),1))
        features = torch.tensor(np.hstack((seq,dseq)), dtype=torch.float32)   
        label = df.loc[df['Timeseries ID'] == idx, 'Label'].values[0]
        
        return features, torch.tensor(label, dtype=torch.long)
    
class MaterialClassifier():
    """
    LSTM-based material classifier for time series temperature data.

    Wraps a PyTorch LSTM model for classification tasks, with support for training,
    evaluation, prediction, and model saving/loading. Works in conjunction with the
    MaterialDataset class for data handling.

    Parameters:
        dataset (MaterialDataset): A MaterialDataset object used for resampling and inference.
        path_name (str): Path to save/load model weights. Defaults to a predefined directory.

    Attributes:
        device (torch.device): GPU or CPU device used for computation.
        lstm (nn.Module): The underlying LSTM model.
        path_name (str): Filesystem path for saving/loading weights.

    Methods:
        load(name): Loads model weights from file.
        save(name): Saves model weights to file.
        predict(timeseries): Predicts the class of a given (temperature, timestamp) series.
        train(training_dataset, validation_dataset, n_epochs, batch_size, lr, gamma):
            Trains the model on the given datasets and returns training/validation losses.
    """
    def __init__(self, dataset:MaterialDataset, path_name = None, num_features=1, num_hidden1=125, num_hidden2=100, num_classes=6):
        """
        Initializes the classifier, sets up the LSTM network and device (CPU or GPU).

        Parameters:
            dataset (MaterialDataset): Dataset manager instance.
            path_name (str, optional): Path to save/load model weights. Defaults to a preset path.
        """
        # Dataset manager
        self.dataset_manager = dataset

        # Check for GPUs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Istanciate LSTM Network
        self.num_features = num_features
        self.lstm = LSTMNet(num_features, num_hidden1, num_hidden2, num_classes=len(self.dataset_manager.class_names))
        self.lstm.to(self.device)
        self.lstm = self.lstm.float()
        self.lstm.eval()

        if path_name is None:
            self.path_name = '/home/leo/Desktop/temperature_ws/src/temperature_ros/config/net/weights/'
        else: 
            self.path_name = path_name

    def load(self, name:str):
        """
        Loads model weights from file.

        Parameters:
            name (str): Name of the weight file (without .pt extension).
        """
        #torch.load(self.path_name + name + '.pt', weights_only=False)
        self.lstm.load_state_dict(torch.load(self.path_name + name + '.pt', weights_only=False))
        self.lstm.eval()

    def save(self, name:str):
        """
        Saves model weights to file.

        Parameters:
            name (str): Name of the weight file (without .pt extension).
        """
        torch.save(self.lstm.state_dict(), self.path_name + name + '.pt')
        print("weight saved at: " + self.path_name + name + '.pt')

    def predict(self, timeseries:np.array):
        """
        Predicts the material class given a raw time series.

        Parameters:
            timeseries (np.array): 2D array with shape (T, N) where timeseries[:,i] are the features
                                   values and timeseries[:,-1] are timestamps (in seconds).

        Returns:
            tuple: (predicted_class, predicted_scores)
        """
        with torch.no_grad():
            t_data = timeseries[:, -1]

            x_data = timeseries[:,0]
            x_data_resampled, _ = self.dataset_manager.resample(x_data, t_data, frequency = 10) # resample at 10Hz
            x_input = x_data_resampled.reshape(len(x_data_resampled),1)
            for i in range(1, self.num_features):
                # Resample to get training rate
                x_data = timeseries[:,i]
                x_data_resampled, _ = self.dataset_manager.resample(x_data, t_data, frequency = 10) # resample at 10Hz
                x_input = np.hstack([x_input, x_data_resampled.reshape(len(x_data_resampled),1)])

            # Tensorize and prepare the input sequence
            x_input = x_input.reshape((1, x_input.shape[0], self.num_features))
            x_input = torch.tensor(x_input, dtype=torch.float32)   
            x_input = x_input.to(self.device)

            # Forward pass (model calls)
            logits_model = self.lstm(x_input)
            # Select the highest value among possible logits
            predicted_scores = F.softmax(logits_model, dim=1).to('cpu')
            predicted_class = torch.argmax(logits_model).to('cpu')
        
            return (predicted_class.numpy(), predicted_scores.numpy()[0])

        #with torch.no_grad():
        #    x_data = timeseries[:,0]
        #    t_data = timeseries[:,1]
        #    x_input, _ = self.dataset_manager.resample(x_data,t_data,frequency = 10) # resample at 10Hz
        #
        #    # Tensorize and prepare the input sequence
        #    x_input = x_input.reshape((1,len(x_input),1))
        #    x_input = torch.tensor(x_input, dtype=torch.float32)   
        #    x_input = x_input.to(self.device)
        #
        #    # Forward pass (model calls)
        #    logits_model = self.lstm(x_input)
        #    # Select the highest value among possible logits
        #    predicted_scores = F.softmax(logits_model, dim=1).to('cpu')
        #    predicted_class = torch.argmax(logits_model).to('cpu')
        #
        #    return (predicted_class.numpy(), predicted_scores.numpy()[0])

    def train(self, training_dataset:MaterialDataset, validation_dataset:MaterialDataset, n_epochs = 100, batch_size = 1, lr=1e-3, gamma=0.9):
        """
        Trains the LSTM model using the provided training and validation datasets.

        Parameters:
            training_dataset (MaterialDataset): Training data.
            validation_dataset (MaterialDataset): Validation data.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Initial learning rate.
            gamma (float): Exponential LR decay factor. Each step lr = gamma*lr 

        Returns:
            dict: Training and validation loss history.
        """
        # Training parameters
        learning_rate = lr
        optimizer = optim.Adam(self.lstm.parameters(), lr = learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        #criterion = nn.NLLLoss()  # for classification with SoftMax included
        loss_fn = nn.CrossEntropyLoss()

        # load dataset
        training_dataloader = DataLoader(training_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         collate_fn=self.collate_fn)
        validation_dataloader = DataLoader(validation_dataset, 
                                           batch_size=1, 
                                           shuffle=False,
                                           collate_fn=self.collate_fn)

        training_loss_list = []
        training_accuracy_list = []
        validation_loss_list = []
        
        # Iterate over the epochs:
        for epoch in range(n_epochs):

            # Set the model to Training mode: This interacts with certain kind of network layers (such as Dropout layers)
            self.lstm.train()  

            # Temporary variable to store the loss on the whole epoch as a convergence metric
            running_loss = 0.0
            training_accuracy = 0.0

            # Iterate on the whole dataset using the dataloader.
            for x_input, lengths, label_gt in training_dataloader:
                # Load inputs ad move to device (GPU)
                x_input, label_gt = x_input.to(self.device), label_gt.to(self.device)
                lengths = lengths.to(self.device)
                #print(x_input)
                # Clear previous gradients
                optimizer.zero_grad()  
                

                # Forward pass (model calls)
                #logits_model = self.lstm(x_input, lengths)  
                # Compute loss (supervised case)
                #loss = loss_fn(logits_model, label_gt)  
                with torch.amp.autocast(str(self.device)):  # if using AMP
                    logits_model = self.lstm(x_input, lengths)
                    loss = loss_fn(logits_model, label_gt)

                # Backpropagation 
                loss.backward()  

                # Update parameters (optimization step)
                optimizer.step()  

                # Update running loss as convergence metric
                running_loss += loss.item()
                
                # update accuracy
                y_pred = torch.argmax(logits_model,dim=1)
                training_accuracy += (y_pred == label_gt).int().to('cpu').numpy().sum()/batch_size

            # Step the learning rate scheduler
            scheduler.step()  

            ## Calculate loss on validation as an additional metric to evaluate overfitting
            # Set the model to Evaluation mode:
            self.lstm.eval()

            # Temporary variable to store the validation loss:
            running_val_loss = 0.0
            
            # Deactivate gradient computation
            with torch.no_grad():
                for x_input, _, label_gt in validation_dataloader:
                    # Load inputs ad move to device (GPU)
                    x_input, label_gt = x_input.to(self.device), label_gt.to(self.device)

                    # Forward pass (model calls)
                    logits_model = self.lstm(x_input)

                    # Compute loss (supervised case)
                    loss = loss_fn(logits_model, label_gt)

                    # Update validation running loss as convergence metric
                    running_val_loss += loss.item()
                    
            # Average epoch loss
            epoch_training_loss = running_loss / len(training_dataloader)
            epoch_training_accuracy = training_accuracy/ len(training_dataloader)*100
            epoch_validation_loss = running_val_loss / len(validation_dataloader)

            # Append the losses to the list:
            training_loss_list.append(epoch_training_loss)
            training_accuracy_list.append(epoch_training_accuracy)
            validation_loss_list.append(epoch_validation_loss)            

            # Convergence metric
            if epoch%50==0:
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1, # Save the next epoch number to start from
                    'model_state_dict': self.lstm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': training_loss_list,
                    'validation loss': validation_loss_list
                }
                torch.save(checkpoint, self.path_name + '/checkpoints/checkpoint_' + datetime.now().strftime("%d%m%Y%H%M%S") + '.pth')

                print(f"[Checkpoint] Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_training_loss:.4f}, Accuracy: {epoch_training_accuracy:.0f}%, Val. Loss: {epoch_validation_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_training_loss:.4f}, Accuracy: {epoch_training_accuracy:.0f}%, Val. Loss: {epoch_validation_loss:.4f}")
            
        results = dict({"loss": training_loss_list, "accuracy": training_accuracy_list, "validation loss": validation_loss_list })
        return results

    
    def collate_fn(self, batch):
        # batch is a list of (sequence, label) tuples
        sequences, labels = zip(*batch)
        
        # Pad sequences to same length
        padded_seqs = pad_sequence(sequences, batch_first=True)  # shape: (batch, max_len, features)
        
        lengths = torch.tensor([len(seq) for seq in sequences])
        labels = torch.stack(labels)

        return padded_seqs, lengths, labels




