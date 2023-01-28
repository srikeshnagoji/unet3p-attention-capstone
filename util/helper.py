import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('../')
from v3_pytorch_refactored.metrics.diceMetrics import dice_coef_metric, compute_iou
from tqdm.notebook import tqdm
import time
import shutil

def pos_neg_diagnosis(mask_path):
    """
    To assign 0 or 1 based on the presence of tumor.
    """
    val = np.max(cv2.imread(mask_path))
    if val > 0: return 1
    else: return 0


def show_aug(inputs, nrows=5, ncols=5, norm=False):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0., hspace=0.)
    i_ = 0
    
    if len(inputs) > 25:
        inputs = inputs[:25]
        
    for idx in range(len(inputs)):
    
        # normalization
        if norm:           
            img = inputs[idx].numpy().transpose(1,2,0)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225] 
            img = (img*std+mean).astype(np.float32)
            
        else:
            img = inputs[idx].numpy().astype(np.float32)
            img = img[0,:,:]
        
        plt.subplot(nrows, ncols, i_+1)
        plt.imshow(img); 
        plt.axis('off')
 
        i_ += 1
        
    return plt.show()

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Previously trained model weights state_dict loaded...')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Previously trained optimizer state_dict loaded...')
    last_epoch = checkpoint['epoch']
    print(f"Previously trained for {last_epoch} number of epochs...")
    return model, optimizer, last_epoch

def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs, device, ckp_path:str=None):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total params of {model_name} model: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params of {model_name} model: {pytorch_total_params}")


    start_epoch=0

    if ckp_path is not None:
        model, optimizer, last_epoch = load_ckp(ckp_path, model, optimizer)
        start_epoch = last_epoch + 1
        print(f"Train for {num_epochs} more epochs...")

    print(f"[INFO] Model is initializing... {model_name}")

    checkpoint_dir = f"/Users/srikeshnagoji/Documents/PythonWorkSpace/jupyter_lab_workspace/PES/CAPSTONE/v3_pytorch_refactored/checkpoints/{model_name}"
    best_model_dir = f"{checkpoint_dir}/{model_name}_best"

    loss_history = []
    train_history = []
    val_history = []
    
    mean_loss_ = 999
    
    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        
        losses = []
        train_iou = []
        
        for i_step, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)
            
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            
            train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            
            loss = train_loss(outputs, target)
            
            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            
        val_mean_iou = compute_iou(model, val_loader, device=device)
        
        mean_loss = np.array(losses).mean()
        loss_history.append(mean_loss)
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.array(losses).mean(),
                    }
        save_ckp(checkpoint, False, checkpoint_dir, best_model_dir)

        if loss<mean_loss_:
            save_ckp(checkpoint, True, checkpoint_dir, best_model_dir)
            mean_loss_ = loss
                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': np.array(losses).mean(),
                #     }, f"{path[:-3]}_best.pt")
        
        

#         print("losses:", np.array(losses))
#         print("iou:", np.array(train_iou))
        print("Epoch [%d]" % (epoch))
        print("Mean loss on train:", np.array(losses).mean(), 
              "\nMean DICE on train:", np.array(train_iou).mean(), 
              "\nMean DICE on validation:", val_mean_iou)
        
    return loss_history, train_history, val_history

def plot_model_history(model_name,
                    train_history, val_history, 
                    num_epochs):

    x = np.arange(num_epochs)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, train_history, label='train dice', lw=3, c="springgreen")
    plt.plot(x, val_history, label='validation dice', lw=3, c="deeppink")

    plt.title(f"{model_name}", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("DICE", fontsize=15)

    fn = str(int(time.time())) + ".png"
    plt.show()

def viz_pred_output(model, loader, idx, test_dataset, device="mps", threshold=0.3):
    valloss = 0
    
    with torch.no_grad():

#         for i_step, (data, target) in enumerate(loader):
        target = torch.tensor(test_dataset[idx][1])
        data = torch.tensor(test_dataset[idx][0])

        data = data.to(device).unsqueeze(0)
        target = target.to(device).unsqueeze(0)

        outputs = model(data)

        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < threshold)] = 0.0
        out_cut[np.nonzero(out_cut >= threshold)] = 1.0

        f, axarr = plt.subplots(1,2)
#             axarr[0,0].imshow(image_datas[0])
#             axarr[0,1].imshow(image_datas[1])

        targ = target.data.cpu().numpy()[0][0]
        target_img = cv2.merge((targ,targ,targ))
        axarr[0].imshow(target_img)

        op = out_cut[0][0]
        axarr[1].imshow(op)



