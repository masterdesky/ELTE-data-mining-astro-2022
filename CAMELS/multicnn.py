from tensorflow.keras import layers as kl
from tensorflow.keras import models as km
from tensorflow.keras import regularizers as kr


class MultiCNN:
    '''
    Creates a CNN model for regression and classification problems with
    multiple types of variables as their outputs. The model supports
    square shaped images.
    
    Parameters
    ----------
    imsize : 
    '''
    def __init__(self,
                 imsize, n_channels,
                 num_filters, kernelsize, padding, stride, kreg,
                 activation):
        self.imsize = imsize
        self.n_channels = n_channels
        self.num_filters = num_filters
        self.kernelsize = kernelsize
        self.padding = padding
        self.stride = stride
        self.kreg = kreg
        self.activation = activation

        self.inputs = kl.Input(shape=(imsize, imsize, n_channels))
        self.branches = []
        
    def __cnn__(self, inputs):
        #
        # Convolutional block 1.
        # 3x3CONVx32 -> 3x3CONVx32 -> MAXPOOLx2
        #
        x = kl.Conv2D(filters=self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(inputs)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))
        
        x = kl.Conv2D(filters=self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))
        
        x = kl.MaxPooling2D(strides=(2, 2))(x)


        #
        # Convolutional block 2.
        # 3x3CONVx64 -> 3x3CONVx64 -> MAXPOOLx2
        #
        x = kl.Conv2D(filters=2*self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=2*self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))
        
        x = kl.MaxPooling2D(strides=(2, 2))(x)


        #
        # Convolutional block 3.
        # 3x3CONVx128 -> 1x1CONVx64 -> 3x3CONVx128 -> MAXPOOLx2
        #
        x = kl.Conv2D(filters=4*self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=2*self.num_filters,
                    kernel_size=(1, 1),
                    padding=self.padding,
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=4*self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.MaxPooling2D(strides=(2, 2))(x)


        #
        # Convolutional block 4.
        # 3x3CONVx256 -> 1x1CONVx128 -> 3x3CONVx256 -> MAXPOOLx2
        #
        x = kl.Conv2D(filters=8*self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=4*self.num_filters,
                    kernel_size=(1, 1),
                    padding=self.padding,
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=8*self.num_filters,
                    kernel_size=(self.kernelsize, self.kernelsize),
                    padding=self.padding,
                    strides=(self.stride, self.stride),
                    kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.MaxPooling2D(strides=(2, 2))(x)


        #
        # Convolutional block 5.
        # 3x3CONVx512 -> 1x1CONVx256 -> 3x3CONVx512 -> AVGPOOL
        #
        x = kl.Conv2D(filters=16*self.num_filters,
                      kernel_size=(self.kernelsize, self.kernelsize),
                      padding=self.padding,
                      strides=(self.stride, self.stride),
                      kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=8*self.num_filters,
                      kernel_size=(1, 1),
                      padding=self.padding,
                      kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))

        x = kl.Conv2D(filters=16*self.num_filters,
                      kernel_size=(self.kernelsize, self.kernelsize),
                      padding=self.padding,
                      strides=(self.stride, self.stride),
                      kernel_regularizer=kr.l2(self.kreg))(x)
        x = kl.Activation(self.activation)(kl.BatchNormalization()(x))
        
        return x
        
    def add_branch(self,
                   n_target, branch_name):
        '''
        Parameters
        ----------
        n_target : int
            Number of target values on this branch. Set `n_target = 1`
            for regression problems, while `n_targets > 1` for
            classification.
        
        branch_name : str
            Arbitrary name given to the branch.
        '''
        assert branch_name is not None, "Every branch should have a name!"
        ## CNN
        x = self.__cnn__(self.inputs)
        # Branch-specific parts
        x = kl.GlobalAveragePooling2D()(x)
        if n_target == 1: activation = None
        else: activation = 'softmax'
        x = kl.Dense(units=n_target, activation=activation,
                     name = f"{branch_name}")(x)
        
        self.branches.append(x);

    def get_model(self):
        assert len(self.branches) > 0, "No target branches added yet!"
        return km.Model(inputs=self.inputs, outputs=self.branches,
                        name="multicnn_net")