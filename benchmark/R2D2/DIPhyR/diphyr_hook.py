from typing import List, Dict, Optional

import numpy as np

class DIPhyRHook:
  def __init__(self, average_window_length=4,):
    self.average_window_length = average_window_length
    self.aw_idxs = [-1]
    self.pred_answers = (-1)*np.ones((self.average_window_length,))
    self.gt_answers = (-1)*np.ones((self.average_window_length,))
    self.perLabel_pred_answers = {}

  def acc_hook(
    self,
    sum_writer,
    env,
    agents,
    env_output_dict,
    obs_count,
    input_streams_dict,
    outputs_stream_dict,
  ):
    logs_dict = input_streams_dict['logs_dict']
    dones = env_output_dict['done']
    info = env_output_dict['succ_info']
    
    nbr_actors = len(dones)
    while len(self.aw_idxs) < nbr_actors:
        self.aw_idxs.append(self.aw_idxs[-1])
    if self.pred_answers.shape[0] != nbr_actors:
        self.pred_answers = (-1)*np.ones((nbr_actors, self.average_window_length))
        self.gt_answers = (-1)*np.ones((nbr_actors,self.average_window_length))
   
    for iidx in range(len(info[0])):
      done = dones[iidx]
      if done:
        self.aw_idxs[iidx] = (self.aw_idxs[iidx]+1) % self.average_window_length
        pred_answer = info[0][iidx]['predicted_answer']
        gt_label = info[0][iidx]['groundtruth_answer']
        if gt_label not in self.perLabel_pred_answers:  self.perLabel_pred_answers[gt_label] = []
        self.perLabel_pred_answers[gt_label].append(pred_answer)
        self.pred_answers[iidx,self.aw_idxs[iidx]] = pred_answer
        self.gt_answers[iidx,self.aw_idxs[iidx]] = gt_label
    
    if not any(dones):  return
    
    acc_buffers = {}
    for akey in list(self.perLabel_pred_answers.keys()):
        lperLabel_pred_answers = self.perLabel_pred_answers[akey]
        if len(lperLabel_pred_answers) > self.average_window_length-1:
            npperLabel_pred_answers = np.asarray(lperLabel_pred_answers)
            gt_answers = akey*np.ones_like(npperLabel_pred_answers)
            acc_buffers[akey] = 100.0*(npperLabel_pred_answers == gt_answers).astype(float)
            while len(lperLabel_pred_answers) >= self.average_window_length:
                del lperLabel_pred_answers[0]
            self.perLabel_pred_answers[akey] = lperLabel_pred_answers

    acc_buffers['Overall'] = 100.0*(self.pred_answers == self.gt_answers).astype(float)
    for akey, acc_buffer in acc_buffers.items():
        values = np.asarray(acc_buffer)
        meanv = values.mean()
        stdv = values.std()
        logs_dict[f"DIPhyR/Accuracy-{akey}/Mean"] = meanv #sum(acc_buffers[s2b_mode])/nbr_actors
        logs_dict[f"DIPhyR/Accuracy-{akey}/Std"] = stdv #sum(acc_buffers[s2b_mode])/nbr_actors
        median_value = np.nanpercentile(
            values,
            q=50,
            axis=None,
            method="nearest"
        )
        q1_value = np.nanpercentile(
            values,
            q=25,
            axis=None,
            method="lower"
        )
        q3_value = np.nanpercentile(
            values,
            q=75,
            axis=None,
            method="higher"
        )
        iqr = q3_value-q1_value
          
        logs_dict[f"DIPhyR/Accuracy-{akey}/Median"] = median_value
        logs_dict[f"DIPhyR/Accuracy-{akey}/Q1"] = q1_value
        logs_dict[f"DIPhyR/Accuracy-{akey}/Q3"] = q3_value
        logs_dict[f"DIPhyR/Accuracy-{akey}/IQR"] = iqr
    
    return   
        

