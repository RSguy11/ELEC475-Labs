# CLIP Training Script for 2.3

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import platform
from CLIP_model_design.clip_pipe import CLIPModel
from Lab4.Load_data_set.load_data_set import create_dataloaders
from torch.amp import autocast, GradScaler

# --------- InfoNCE Loss for CLIP ---------
def clip_infonce_loss(logits):
	# logits: [batch_size, batch_size] similarity matrix
	labels = torch.arange(logits.size(0)).to(logits.device)
	loss_i2t = nn.CrossEntropyLoss()(logits, labels)
	loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
	return (loss_i2t + loss_t2i) / 2

# --------- Training Function ---------

def train_clip(
	epochs=5,
	batch_size=8,
	lr=1e-4,
	device=None,
	max_samples=100,
	data_root=None
):
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = CLIPModel().to(device)
	optimizer = optim.AdamW([
		{"params": model.image_encoder.parameters()},
		{"params": model.text_encoder.parameters(), "lr": 0.0},  # frozen
	], lr=lr)
	# Add learning rate scheduler
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
	scaler = GradScaler('cuda')  # For mixed precision
	# Set the correct COCO2014 path
	if data_root is None:
		data_root = "../coco2014"  # relative to this script, adjust if needed
	train_loader, val_loader = create_dataloaders(data_root=data_root, batch_size=batch_size, max_samples=max_samples)
	train_losses, val_losses = [], []
	start_time = time.time()

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		for epoch in range(epochs):
			model.train()
			running_loss = 0.0
			for batch in train_loader:
				images = batch['image'].to(device)
				texts = batch['text']  # list of strings
				optimizer.zero_grad()
				with autocast('cuda'):
					out = model(images, texts)
					loss = clip_infonce_loss(out['logits'])
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				running_loss += loss.item() * images.size(0)
			avg_train_loss = running_loss / len(train_loader.dataset)
			train_losses.append(avg_train_loss)

			# Validation
			model.eval()
			val_loss = 0.0
			with torch.no_grad():
				for batch in val_loader:
					images = batch['image'].to(device)
					texts = batch['text']
					with autocast('cuda'):
						out = model(images, texts)
						loss = clip_infonce_loss(out['logits'])
					val_loss += loss.item() * images.size(0)
			avg_val_loss = val_loss / len(val_loader.dataset)
			val_losses.append(avg_val_loss)

			# Step the scheduler with validation loss
			scheduler.step(avg_val_loss)

			print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

			# Save checkpoint
			checkpoint_path = f"clip_checkpoint_epoch{epoch+1}.pth"
			torch.save({
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'train_loss': avg_train_loss,
				'val_loss': avg_val_loss
			}, checkpoint_path)
			print(f"Checkpoint saved: {checkpoint_path}")
	return train_losses, val_losses, total_time, device, model

# --------- Plotting and Reporting ---------
def plot_losses(train_losses, val_losses):
	plt.figure()
	plt.plot(train_losses, label='Train Loss')
	plt.plot(val_losses, label='Val Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('CLIP Training/Validation Loss Curves')
	plt.savefig('clip_loss_curves.png')
	plt.close()

def report(total_time, device):
	print("\n--- Training Report ---")
	print(f"Total training time: {total_time:.2f} seconds")
	print(f"Hardware used: {device} ({platform.platform()})")
	print("Observed issues: ")
	print("- If loss diverges or is unstable, try lowering the learning rate or using gradient clipping.")
	print("- If training is slow, use a smaller batch size or image resolution.")
	print("- If you see CUDA OOM errors, reduce batch size.")

# --------- Main ---------
if __name__ == "__main__":
	# Set the correct path to coco2014 (relative to this script)
	coco2014_path = "../coco2014"  # adjust if needed
	train_losses, val_losses, total_time, device, model = train_clip(
		epochs=5, batch_size=16, lr=1e-4, max_samples=100, data_root=coco2014_path
	)
	plot_losses(train_losses, val_losses)
	report(total_time, device)
	print("Loss curves saved as clip_loss_curves.png")
