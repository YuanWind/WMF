# Wind Model Frame

│  config.cfg				配置文件  
│  Config.py				配置文件类  
│  main.py					主函数  
│  README.md				说明  
│  run.sh					运行脚本  
│  try.py					试错脚本: 测试一些自己不确定的代码  
│    
├─data  
│  │  DataProcess.py		数据处理脚本  
│          
├─modules  
│      Model.py				模型类  
│      Optimizer.py			优化器类  
│        
├─saved						保存做完实验的配置文件和日志，以便进行对比结果  
├─saved_models				保存模型  
├─tmp						保存生成的临时文件  
└─utils  
    │  Dataloader.py		定义样本类和词表类  
    │  evaluate.py			评测脚本  
    │  Trainer.py			训练脚本	  
    │  utils.py				通用的一些函数  