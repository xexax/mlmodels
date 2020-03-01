from transformers.modeling_gpt2 import GPT2LMHeadModel
from pplm.run_pplm import run_pplm_example

####################################################################################################
class Model:
    def __init__(self):
        # 4.Define Model    
        # This downloads GPT-2 Medium, it takes a little while
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        
 



def predict(cond_text,bag_of_words,discrim=None,class_label=-1):
    print(" Generating text ... ")
    unpert_gen_text, pert_gen_text = run_pplm_example(
                        cond_text=cond_text,
                        num_samples=3,
                        bag_of_words=bag_of_words,
                        length=50,
                        discrim=discrim,
                        class_label=class_label,
                        stepsize=0.03,
                        sample=True,
                        num_iterations=3,
                        window_length=5,
                        gamma=1.5,
                        gm_scale=0.95,
                        kl_scale=0.01,
                        verbosity="quiet"
                    )
    print(" Unperturbed generated text :\n")
    print(unpert_gen_text)
    print()
    print(" Perturbed generated text :\n")
    print(pert_gen_text)
    print()
                    
if __name__ == '__main__':
    # initializing the model
    model = Model()
    # generating teh text
    predict(cond_text="The potato",bag_of_words='military')