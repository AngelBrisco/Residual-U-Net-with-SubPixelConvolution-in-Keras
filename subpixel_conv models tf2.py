import tensorflow as tf


class SubpixelLayer2D(Layer):

    def __init__(self,filters=None,ksz=1, scale=2, **kwargs):
        self.scale=scale
        self.out_channels=filters
        self.ksz=ksz
        
        super(SubpixelLayer2D, self).__init__(**kwargs)
    
    def kinit(self,shape,dtype=None,partition_info=None):
        h,w,cin,cout=shape
        
        kw=tf.keras.initializers.RandomNormal()(shape=(h,w,cin,self.out_channels))
        kwn=tf.tile(kw,[1,1,1,self.scale**2])
        
        sp_weights=tf.Variable(kwn,
                            dtype=dtype,
                            name="kernel")
        return sp_weights
    
    def build(self, input_shape):
        b,h,w,cin=input_shape
        if self.out_channels==None:
            self.out_channels=(cin)//(self.scale**2)
        
        self.kernel = self.add_weight(shape=(self.ksz,self.ksz,cin,self.out_channels*self.scale**2),
                                      initializer=self.kinit,
                                      name='kernel')

        super(SubpixelLayer2D, self).build(input_shape)

    def call(self,input):
        y = tf.keras.backend.conv2d(input, self.kernel, strides=(1, 1), padding='same', data_format="channels_last",
                                    dilation_rate=(1, 1))
        y = tf.keras.backend.relu(y)
        y = tf.nn.depth_to_space(y, self.scale)
        y = tf.keras.backend.pool2d(y, pool_size=(self.scale,self.scale), strides=(1, 1),  padding='same',  data_format="channels_last", pool_mode='avg')
        return y

    def compute_output_shape(self,input_shape):
        shape=input_shape
        return(shape[0],
               shape[1] * self.scale,
               shape[2] * self.scale,
               self.out_channels)
    
    def get_config(self):
        base_config = super(SubpixelLayer2D, self).get_config()
        base_config['filters'] = self.out_channels
        base_config['scale'] = self.scale
        base_config['ksz'] = self.ksz
        return base_config






class SubpixelLayer2D_log2(Layer):

    def __init__(self,filters=None,ksz=1, scale=2, **kwargs):
        self.loop=int(np.log2(scale))-1
        self.scale=scale
        self.prime_scale=2
        self.out_channels=filters
        self.ksz=ksz
        self.loop_kernel={}

        super(SubpixelLayer2D_log2, self).__init__(**kwargs)

    def kinit(self,shape,dtype=None,partition_info=None):
        h,w,cin,cout=shape

        #Multiplica el kernel para evitar efecto tablero. Aunque no lo creas lo entendiste
        y=tf.keras.initializers.RandomNormal()(shape=(h,w,cin,cout//self.prime_scale**2))
        y=tf.tile(y,[1,1,1,self.prime_scale**2])

        sp_weights=tf.Variable(y,
                            dtype=dtype,
                            name="kernel")
        return sp_weights

    def build(self, input_shape):
        b,h,w,cin=input_shape
        if self.out_channels==None:
            self.out_channels=cin

        self.kernel = self.add_weight(shape=(self.ksz,self.ksz,cin,self.out_channels*self.prime_scale**2),
                                      initializer=self.kinit,
                                      name='kernel')
        for i in range(self.loop):
            self.loop_kernel[i] = self.add_weight(shape=(self.ksz,self.ksz,self.out_channels,self.out_channels*self.prime_scale**2),
                                          initializer=self.kinit,
                                          name='loop_kernel%d'%i)

        super(SubpixelLayer2D_log2, self).build(input_shape)

    def call(self,input):
        for i in range(self.loop+1):
            kernel=self.kernel if i==0 else self.loop_kernel[i-1]
            x= input if i==0 else y
            y = tf.keras.backend.conv2d(x, kernel, strides=(1, 1), padding='same', data_format="channels_last",
                                        dilation_rate=(1, 1))
            y = tf.nn.depth_to_space(y, self.prime_scale)
            y = tf.keras.backend.relu(y)
            y = tf.keras.backend.pool2d(y, pool_size=(self.prime_scale,self.prime_scale), strides=(1, 1),  padding='same',  data_format="channels_last", pool_mode='avg')
        return y

    def compute_output_shape(self,input_shape):
        shape=input_shape
        return(shape[0],
               shape[1] * self.scale,
               shape[2] * self.scale,
               self.out_channels)

    def get_config(self):
        base_config = super(SubpixelLayer2D_log2, self).get_config()
        base_config['filters'] = self.out_channels
        base_config['scale'] = self.scale
        base_config['ksz'] = self.ksz
        return base_config
