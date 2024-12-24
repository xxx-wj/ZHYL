#模型架构选择与修改:
model = models.efficientnet_v2_s(weights='DEFAULT')
model.classifier[1] = torch.nn.Linear(1280, class_size)

#损失函数:
criterion = nn.CrossEntropyLoss()

#优化器设置:
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#学习率调度器:
lr_milestones = [1000] #只训练十遍，不需要学习率调度器，所以设置为1000
multi_step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

#训练过程中的超参数:
EPOCHS = 10
learning_rate = 0.0001
momentum = 0.9
weight_decay = 0.1

#要查看或保存这些参数的具体数值
#可以通过 torch.save(model.state_dict()"model_parameters.pth") 的方式保存模型状态字典
#其中包含了模型所有可学习参数的当前值。模型在训练过程中的最佳状态已经保存为"checkpoints/best.pth"
