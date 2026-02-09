from keras.layers import Dense, Activation, Dropout, Bidirectional, LSTM
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback
import os
import numpy as np
import gc
import json
import time
import signal
import sys
import shutil
np.set_printoptions(suppress=True)

# Import centralized configuration
try:
    import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("⚠️ Warning: config.py not found, using default values")

from vgg16_feature_extractor import extract_vgg16_features_live, scan_and_extract_vgg16_features

# Default batch size for large datasets
DEFAULT_BATCH_SIZE = config.BATCH_SIZE if USE_CONFIG else 625
# Minimum batch size for small datasets
MIN_BATCH_SIZE = 2

# Use config.py values if available
NUM_EPOCHS = config.EPOCHS if USE_CONFIG else 100
VERBOSE = 1
HIDDEN_UNITS = config.LSTM_UNITS if USE_CONFIG else 512
MAX_ALLOWED_FRAMES = 125
EMBEDDING_SIZE = 100

K.set_image_data_format('channels_last')


class TrainingStateManager:
    """Manages training state for resume capability"""
    
    def __init__(self, model_dir_path):
        self.state_file = os.path.join(model_dir_path, 'training_state.json')
        self.checkpoint_dir = os.path.join(model_dir_path, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_state(self, epoch, total_epochs, best_loss, history_dict):
        """Save current training state"""
        state = {
            'current_epoch': epoch,
            'total_epochs': total_epochs,
            'best_val_loss': float(best_loss) if best_loss else None,
            'completed': epoch >= total_epochs,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'history': {
                'loss': [float(x) for x in history_dict.get('loss', [])] if history_dict.get('loss') else [],
                'val_loss': [float(x) for x in history_dict.get('val_loss', [])] if history_dict.get('val_loss') else [],
                'accuracy': [float(x) for x in history_dict.get('accuracy', [])] if history_dict.get('accuracy') else [],
                'val_accuracy': [float(x) for x in history_dict.get('val_accuracy', [])] if history_dict.get('val_accuracy') else []
            }
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        best_loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        print(f"[State] Saved: Epoch {epoch}/{total_epochs}, Best loss: {best_loss_str}")
    
    def load_state(self):
        """Load previous training state"""
        if not os.path.exists(self.state_file):
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception as e:
            print(f"Warning: Could not load training state: {e}")
            return None
    
    def get_resume_info(self):
        """Get information about resumable training"""
        state = self.load_state()
        if not state:
            return None
        
        info = {
            'can_resume': True,
            'completed_epochs': state.get('current_epoch', 0),
            'total_epochs': state.get('total_epochs', 100),
            'best_val_loss': state.get('best_val_loss'),
            'is_complete': state.get('completed', False),
            'last_updated': state.get('timestamp'),
            'remaining_epochs': state.get('total_epochs', 100) - state.get('current_epoch', 0)
        }
        
        return info


class EpochStateCallback(Callback):
    """Callback to save training state after each epoch"""
    
    def __init__(self, state_manager, total_epochs):
        super().__init__()
        self.state_manager = state_manager
        self.total_epochs = total_epochs
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Track best loss
        val_loss = logs.get('val_loss')
        if val_loss and val_loss < self.best_loss:
            self.best_loss = val_loss
        
        # Save state
        self.state_manager.save_state(
            epoch=epoch + 1,  # Keras epochs are 0-indexed
            total_epochs=self.total_epochs,
            best_loss=self.best_loss,
            history_dict=self.model.history.history
        )


class InterruptHandler:
    """Handle Ctrl+C gracefully"""
    
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        if not self.interrupted:
            print("\n" + "="*70)
            print("TRAINING INTERRUPTED (Ctrl+C)")
            print("="*70)
            print("Saving current state before exit...")
            print("You can resume training later from the latest checkpoint.")
            print("Run: python train_asd_model.py")
            print("="*70)
            self.interrupted = True
        else:
            print("\nForce quitting...")
            sys.exit(1)


def get_adaptive_batch_size(num_samples):
    """
    Calculate adaptive batch size based on dataset size.
    For small datasets (< 100 samples), use smaller batches.
    """
    if num_samples < 10:
        return MIN_BATCH_SIZE
    elif num_samples < 20:
        return min(MIN_BATCH_SIZE, num_samples // 2)
    elif num_samples < 50:
        return min(4, num_samples // 2)
    elif num_samples < 100:
        return min(8, num_samples // 2)
    elif num_samples < 500:
        return min(32, num_samples // 4)
    else:
        return min(DEFAULT_BATCH_SIZE, num_samples // 4)


def generate_batch(x_samples, y_samples, batch_size=None):
    """
    Generate batches for training/validation.
    If batch_size is None, it will be calculated adaptively.
    """
    if batch_size is None:
        batch_size = get_adaptive_batch_size(len(x_samples))
    
    num_batches = max(1, len(x_samples) // batch_size)  # Ensure at least 1 batch
    
    # Printing the details about the input to the training model
    #print("Number of X Samples: " + str(len(x_samples)))
    #print("Number of Y Samples: " + str(len(y_samples)))
    #print(f"Batch size: {batch_size}")
    #print("Number of Batches: " + str(num_batches))

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = min((batchIdx + 1) * batch_size, len(x_samples))  # Don't exceed array bounds
            yield np.array(x_samples[start:end]), y_samples[start:end]


class vgg16BidirectionalLSTMVideoClassifier(object):
    model_name = 'vgg16-bidirectional-lstm'

    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.vgg16_model = None
        self.expected_frames = None
        self.vgg16_include_top = True
        self.config = None

    def create_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True),
                                input_shape=(self.expected_frames, self.num_input_tokens)))
        model.add(Bidirectional(LSTM(10)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.nb_classes))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    @staticmethod
    def get_config_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + vgg16BidirectionalLSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + vgg16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + vgg16BidirectionalLSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + vgg16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + vgg16BidirectionalLSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + vgg16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def load_model(self, config_file_path, weight_file_path):
        if os.path.exists(config_file_path):
            print('loading configuration from ', config_file_path)
        else:
            raise ValueError('cannot locate config file {}'.format(config_file_path))

        config = np.load(config_file_path, allow_pickle = True).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.vgg16_include_top = config['vgg16_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])
        self.config = config

        self.model = self.create_model()
        if os.path.exists(weight_file_path):
            print('loading network weights from ', weight_file_path)
        else:
            raise ValueError('cannot local weight file {}'.format(weight_file_path))

        self.model.load_weights(weight_file_path)

        print('build vgg16 with pre-trained model')
        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp

        predicted_c = self.model.predict(np.array([x]), verbose=0)[0]
        predicted_class = np.argmax(predicted_c)
        predicted_label = self.labels_idx2word[predicted_class]
        print('predicted_prob is: '+ str(predicted_c))
        return predicted_label
    
    def predict_with_confidence(self, video_file_path):
        """Predict class with confidence scores"""
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_c = self.model.predict(np.array([x]), verbose=0)[0]
        return predicted_c

    def fit(self, data_dir_path, model_dir_path, vgg16_include_top=True, data_set_name='autism_data', test_size=0.2,
            random_state=42):

        self.vgg16_include_top = vgg16_include_top

        config_file_path = self.get_config_file_path(model_dir_path, vgg16_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, vgg16_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, vgg16_include_top)

        self.vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        self.vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        feature_dir_name = data_set_name + '-vgg16-Features'
        if not vgg16_include_top:
            feature_dir_name = data_set_name + '-vgg16-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.vgg16_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
        self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print(self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['vgg16_include_top'] = self.vgg16_include_top

        self.config = config

        np.save(config_file_path, config)

        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)

        # Calculate adaptive batch size
        train_batch_size = get_adaptive_batch_size(len(Xtrain))
        test_batch_size = get_adaptive_batch_size(len(Xtest))
        
        print(f"\nDataset size: {len(x_samples)} samples")
        print(f"Training samples: {len(Xtrain)}, batch size: {train_batch_size}")
        print(f"Testing samples: {len(Xtest)}, batch size: {test_batch_size}")

        train_gen = generate_batch(Xtrain, Ytrain, train_batch_size)
        test_gen = generate_batch(Xtest, Ytest, test_batch_size)

        train_num_batches = max(1, len(Xtrain) // train_batch_size)
        test_num_batches = max(1, len(Xtest) // test_batch_size)
        #Printing the number of batches for testing and training depending upon batch size and test train split
        print(f"Train batches: {train_num_batches}")
        print(f"Test batches: {test_num_batches}\n")

        # Initialize training state manager
        state_manager = TrainingStateManager(model_dir_path)
        interrupt_handler = InterruptHandler(state_manager)
        
        # Create checkpoints directory
        checkpoint_dir = os.path.join(model_dir_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callback 1: Save best model (monitored by validation loss)
        best_checkpoint = ModelCheckpoint(
            filepath=weight_file_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1,
            save_weights_only=False
        )
        
        # Callback 2: Save every 10 epochs (periodic backup)
        periodic_checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'epoch-{epoch:03d}-loss-{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=False,
            period=10,
            verbose=1,
            save_weights_only=False
        )
        
        # Callback 3: Save latest checkpoint (for resume)
        latest_checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'latest_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=False,
            period=1,
            verbose=0,
            save_weights_only=False
        )
        
        # Callback 4: Save training state
        state_callback = EpochStateCallback(state_manager, NUM_EPOCHS)
        
        # Use all callbacks
        callbacks = [best_checkpoint, periodic_checkpoint, latest_checkpoint, state_callback]
        
        print("\n" + "="*70)
        print(f"STARTING TRAINING - {NUM_EPOCHS} EPOCHS")
        print("="*70)
        print(f"📊 Progress will be saved automatically")
        print(f"📂 Checkpoints: {checkpoint_dir}")
        print("="*70 + "\n")
        
        history = model.fit(train_gen, steps_per_epoch=train_num_batches,
                           epochs=NUM_EPOCHS,
                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                           callbacks=callbacks)
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"📂 Model saved to: {weight_file_path}")
        print(f"📂 Checkpoints saved in: {checkpoint_dir}")
        print("="*70 + "\n")

        return history


class vgg16LSTMVideoClassifier(object):
    model_name = 'vgg16-lstm'
    
    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.vgg16_model = None
        self.expected_frames = None
        self.vgg16_include_top = None
        self.config = None
        
    @staticmethod
    def get_config_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + vgg16LSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + vgg16LSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + vgg16LSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + vgg16LSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + vgg16LSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + vgg16LSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def create_model(self):
        model = Sequential()

        model.add(
            LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def load_model(self, config_file_path, weight_file_path):

        config = np.load(config_file_path, allow_pickle = True).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.vgg16_include_top = config['vgg16_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])

        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_c = self.model.predict(np.array([x]), verbose=0)[0]
        predicted_class = np.argmax(predicted_c)
        predicted_label = self.labels_idx2word[predicted_class]
        print('predicted prob is: '+ str(predicted_c))
        return predicted_label
    
    def predict_with_confidence(self, video_file_path):
        """Predict class with confidence scores"""
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_c = self.model.predict(np.array([x]), verbose=0)[0]
        return predicted_c

    def fit(self, data_dir_path, model_dir_path, vgg16_include_top=True, data_set_name='autism_data', test_size=0.2, random_state=42):
        self.vgg16_include_top = vgg16_include_top

        config_file_path = self.get_config_file_path(model_dir_path, vgg16_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, vgg16_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, vgg16_include_top)

        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

        feature_dir_name = data_set_name + '-vgg16-Features'
        if not vgg16_include_top:
            feature_dir_name = data_set_name + '-vgg16-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.vgg16_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
            self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            print(x.shape)
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print(self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)
        y_samples = to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['vgg16_include_top'] = self.vgg16_include_top
        self.config = config

        np.save(config_file_path, config)

        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)

        # Calculate adaptive batch size
        train_batch_size = get_adaptive_batch_size(len(Xtrain))
        test_batch_size = get_adaptive_batch_size(len(Xtest))
        
        print(f"\nDataset size: {len(x_samples)} samples")
        print(f"Training samples: {len(Xtrain)}, batch size: {train_batch_size}")
        print(f"Testing samples: {len(Xtest)}, batch size: {test_batch_size}")

        train_gen = generate_batch(Xtrain, Ytrain, train_batch_size)
        test_gen = generate_batch(Xtest, Ytest, test_batch_size)

        train_num_batches = max(1, len(Xtrain) // train_batch_size)
        test_num_batches = max(1, len(Xtest) // test_batch_size)
        #print("Number of train batches: " + str(train_num_batches))
        #print("Number of test batches: " + str(test_num_batches))
        print(f"Train batches: {train_num_batches}")
        print(f"Test batches: {test_num_batches}\n")

        # Initialize training state manager
        state_manager = TrainingStateManager(model_dir_path)
        interrupt_handler = InterruptHandler(state_manager)
        
        # Create checkpoints directory (use config.py path if available)
        if USE_CONFIG and hasattr(config, 'CHECKPOINTS_PATH'):
            checkpoint_dir = config.CHECKPOINTS_PATH
        else:
            checkpoint_dir = os.path.join(model_dir_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Display checkpoint configuration
        print("\n" + "="*70)
        print("CHECKPOINT CONFIGURATION")
        print("="*70)
        print(f"✓ Checkpoints enabled")
        print(f"  Directory: {checkpoint_dir}")
        print(f"  Best model: {weight_file_path}")
        print(f"  Periodic saves: Every 10 epochs")
        print(f"  Latest checkpoint: Every epoch (for resume)")
        print(f"\n💡 Training can be interrupted (Ctrl+C) and resumed later")
        print("="*70 + "\n")
        
        # Callback 1: Save best model (monitored by validation loss)
        best_checkpoint = ModelCheckpoint(
            filepath=weight_file_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1,
            save_weights_only=False
        )
        
        # Callback 2: Save every 10 epochs (periodic backup)
        periodic_checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'epoch-{epoch:03d}-loss-{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=False,
            period=10,
            verbose=1,
            save_weights_only=False
        )
        
        # Callback 3: Save latest checkpoint (for resume)
        latest_checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'latest_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=False,
            period=1,
            verbose=0,
            save_weights_only=False
        )
        
        # Callback 4: Save training state
        state_callback = EpochStateCallback(state_manager, NUM_EPOCHS)
        
        # Use all callbacks
        callbacks = [best_checkpoint, periodic_checkpoint, latest_checkpoint, state_callback]
        
        print("\n" + "="*70)
        print(f"STARTING TRAINING - {NUM_EPOCHS} EPOCHS")
        print("="*70)
        print(f"📊 Progress will be saved automatically")
        print(f"📂 Checkpoints: {checkpoint_dir}")
        print("="*70 + "\n")
        
        history = model.fit(train_gen, steps_per_epoch=train_num_batches,
                           epochs=NUM_EPOCHS,
                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                           callbacks=callbacks)
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"📂 Model saved to: {weight_file_path}")
        print(f"📂 Checkpoints saved in: {checkpoint_dir}")
        print("="*70 + "\n")

        return history
