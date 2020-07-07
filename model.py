import tensorflow as tf

class deepFM(tf.keras.Model):
    def __init__(self,parameters):
        super().__init__()
        
        #cols used in input dataset
        self.fm_cols=parameters['fm_cols']
        
        #embedding dimension
        self.fm_emb_dim=parameters['fm_emb_dim']
        
        #hidden layers structure
        self.hidden_units=parameters['hidden_units']
        
        #dropout probability
        self.dropprob=parameters['dropprob']

        with tf.name_scope('Embedding'):
            self.fm_emb=tf.Variable(tf.random.normal([len(self.fm_cols),self.fm_emb_dim],0,0.01),
                                name='fm_embed_matrix')

        with tf.name_scope('DNN'):
            self.hidden_layers=[]
            for i,unit in enumerate(self.hidden_units):
                self.hidden_layers+=[
                    tf.keras.layers.Dense(unit,activation=tf.nn.relu,name='dnn_layer_%d'%i),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(rate=self.dropprob)
                ]

        with tf.name_scope('output'):
            self.final_bn=tf.keras.layers.BatchNormalization()
            self.final_do=tf.keras.layers.Dropout(rate=self.dropprob)
            self.final_output=tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,name='output_layer')

    def call(self,inputs,training=False):
        # -------low order ----------
        fm_col_vals = tf.Variable(
            tf.zeros((len(inputs[self.fm_cols[0]]), len(self.fm_cols))), trainable=False ) # [B,F]

        for i in range(len(self.fm_cols)):
            fm_col_vals[:,i].assign(tf.cast(inputs[self.fm_cols[i]],dtype=tf.float32)) #[B,1]

            fm_col_emb=tf.tile(tf.gather(self.fm_emb,[i],axis=0),[len(inputs[self.fm_cols[0]]),1]) #[B,H]
            fm_col_emb=fm_col_emb*tf.expand_dims(fm_col_vals[:,i],axis=1) #[B,H]
            if i==0:
                fm_col_embs=tf.expand_dims(fm_col_emb,axis=1) #[B,1,H]
            else:
                fm_col_embs=tf.concat([fm_col_embs,tf.expand_dims(fm_col_emb,axis=1)],axis=1)

        summed_ft_emb=tf.reduce_sum(fm_col_embs,axis=1) #[B,H]
        summed_ft_emb_square=tf.square(summed_ft_emb) #[B,H]

        squared_ft_emb=tf.square(fm_col_embs) #[B,F,H]
        squared_ft_emb_sum = tf.reduce_sum(squared_ft_emb, axis=1)  # [B,H]

        second_orders=0.5*tf.subtract(summed_ft_emb_square,squared_ft_emb_sum) # [B,H]

        # -------high order ----------
        high_orders=tf.reshape(fm_col_embs,[-1,len(self.fm_cols)*self.fm_emb_dim])
        for i in range(len(self.hidden_layers)//3):
            high_orders=self.hidden_layers[3*i](high_orders)
            high_orders = self.hidden_layers[3*i+1](high_orders)
            high_orders = self.hidden_layers[3*i+2](high_orders,training=training)

        all_i=tf.concat([fm_col_vals,second_orders,high_orders],axis=1)
        all_i=self.final_bn(all_i)
        all_i=self.final_do(all_i)
        return self.final_output(all_i)