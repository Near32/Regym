from typing import List, Dict, Optional


class DIPhyRHook:
  def __init__(self, average_window_length=4,):
    self.average_window_length = average_window_length
    self.aw_idx = -1
    self.pred_answers = np.zeros((self.average_window_length,))
    self.gt_answers = np.ones((self.average_window_length,))
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
    
    import ipdb; ipdb.set_trace()
    for iidx in range(len(info[0])):
      done = done[iidx]
      if done:
        self.aw_idx = (self.aw_idx+1) % self.average_window_length
        pred_answer = info['predicted_answer']
        gt_label = info['groundtruth_answer']
        if gt_label not in self.perLabel_pred_answers:  self.perLabel_pred_answers[gt_label] = []
        self.perLabel_pred_answers[gt_label].append(pred_answer)
        self.pred_answers[self.aw_idx] = pred_answer
        self.gt_answers[self.aw_idx] = gt_label
    
    acc_buffers = {}
    for akey, lperLabel_pred_answers in self.perLabel_pred_answers.items():
        if len(lperLabel_pred_answers) >= self.average_window_length:
            import ipdb; ipdb.set_trace()
            npperLabel_pred_answers = np.asarray(lperLabel_pred_answers)
            gt_answers = akey*np.ones_like(npperLabel_pred_answers)
            acc_buffers[akey] = 100.0*(npperLabel_pred_answers == gt_answers).float()
    
    acc_buffers['Overall'] = 100.0*(self.pred_answers == self.gt_answers).float()
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
        

