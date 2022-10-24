import logging
import datetime as dt
import os


class BaseModel:
    def __init__(self, name="unknown"):
        self.name = name

    def __str__(self):
        return f'<{self.name}>'

    def save_model(self, base_dir, model, model_eval_hist):
        logging.debug(f'SAVING MODEL TO {base_dir}')
        date_time_format = '%Y_%m_%d__%H_%M_%S'
        current_date_time_dt = dt.datetime.now()
        current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
        model_evaluation_loss, model_evaluation_accuracy = model_eval_hist
        model_name = os.path.join(base_dir, f'Model__Date_Time_{current_date_time_string}'
                                            f'___Loss_{model_evaluation_loss}'
                                            f'___Accuracy_{model_evaluation_accuracy}.h5'
                                  )
        # Saving your Model
        model.save(model_name)
