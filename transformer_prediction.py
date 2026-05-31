import math
import numpy as np
import keras
import tensorflow as tf
import transformer_seq2seq



class Predictor(tf.Module):
  def __init__(self, ):
    self.transformer_s2s = transformer_seq2seq.Transformer()

  def __call__(self, input, pred_size, start):

    encoder_input = input

    output_array = np.zeros((1,pred_size,1),dtype=float)
    output_array[0,0,0] = start


    for i in tf.range(pred_size):
      output = output_array
      prediction = self.transformer_s2s([encoder_input, output])
      output_array[0,i+1,0] = prediction[:,0,:]


    output = output_array
    return output

