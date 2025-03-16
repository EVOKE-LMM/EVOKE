pip install -r requirements.txt
pip install -r requirements_openai_api.txt
pip install -r requirements_web_demo.txt

pip install peft
pip install optimum
pip install auto-gptq


pip install transformers==4.36.2
pip install accelerate==0.34.2

pip uninstall gradio numpy lit triton
pip install gradio==5.15.0                    
pip install numpy==1.26.4
pip install lit==15.0.7
pip install triton==2.0.0                     
pip install deepspeed==0.12.4