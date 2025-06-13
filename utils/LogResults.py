class LogResults():
    def __init__(self, error_name:str, output_directory:str, loss_fxn_name:str):
        """
        error_types is train, val, or test
        """
        self.output_directory = output_directory
        self.composite_loss = False
        self.error_name = error_name
        self.types = ["jsd", 
                    "correlation", 
                    "bp_correlation",
                    "ocr_jsd",
                    "ocr_bp_correlation",
                    "top_ocr_jsd"]
        
        if (loss_fxn_name == "CompositeLoss" or "CompositeLossBalanced"):
            self.losses = ["error", "scalar_error", "profile_error"]



    def record_results(self, epoch, dataset_len:float, n_batches:int, loss:float, jsd:float, correlation:float, bp_correlation:float, 
                        ocr_jsd:float, ocr_bp_correlation:float, top_ocr_jsd:float,
                        scalar_error:float=None, profile_error:float=None):
        # errors = [loss, jsd, correlation, bp_correlation, ocr_correlation, scalar_error, profile_error]
        metrics = [jsd, correlation, bp_correlation, ocr_jsd, ocr_bp_correlation, top_ocr_jsd]
        errors = [loss, scalar_error, profile_error]

        for i in range(0, len(self.types)):
            with open(self.output_directory + "/" + self.error_name + "_" + self.types[i] + ".txt", "a") as f:
                print("%s, %s" % (epoch, metrics[i]/dataset_len), file=f, flush=True)
        
        for i in range(0, len(self.losses)):
            with open(self.output_directory + "/" + self.error_name + "_" + self.losses[i] + ".txt", "a") as f:
                print("%s, %s" % (epoch, errors[i]/n_batches), file=f, flush=True)

