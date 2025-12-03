# CLIP Training Script for 2.3
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import platform
from CLIP_model_design.clip_pipe import CLIPModel
from Load_data_set.load_data_set import create_dataloaders
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --------- InfoNCE Loss for CLIP ---------
def clip_infonce_loss(logits, label_smoothing=0.1):
	# logits: [batch_size, batch_size] similarity matrix
	labels = torch.arange(logits.size(0)).to(logits.device)
	# Label smoothing for better generalization
	loss_i2t = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels)
	loss_t2i = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits.t(), labels)
	return (loss_i2t + loss_t2i) / 2

# --------- Training Function ---------

def compute_cosine_similarity_matrix(image_embeds, text_embeds):
	image_embeds = F.normalize(image_embeds, p=2, dim=1)
	text_embeds = F.normalize(text_embeds, p=2, dim=1)
	return image_embeds @ text_embeds.T

def recall_at_k(sim_matrix, k=1):
	N = sim_matrix.size(0)
	# Image to Text recall
	i2t_ranks = torch.argsort(sim_matrix, dim=1, descending=True)
	i2t_correct = torch.arange(N).unsqueeze(1).expand(-1, k) == i2t_ranks[:, :k]
	recall_i2t = i2t_correct.any(dim=1).float().mean().item()
	# Text to Image recall  
	t2i_ranks = torch.argsort(sim_matrix, dim=0, descending=True)
	t2i_correct = torch.arange(N).unsqueeze(0).expand(k, -1) == t2i_ranks[:k, :]
	recall_t2i = t2i_correct.any(dim=0).float().mean().item()
	return recall_i2t, recall_t2i

def train_clip(
	epochs=5,
	batch_size=8,
	lr=1e-4,
	device=None,
	max_samples=1000,
	data_root=None
):
	print("max samples:", max_samples)
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = CLIPModel().to(device)
	# Exponential Moving Average for better validation performance
	ema_model = CLIPModel().to(device)
	ema_model.load_state_dict(model.state_dict())
	ema_decay = 0.999
	optimizer = optim.AdamW([
		{"params": model.image_encoder.vision_model.parameters(), "lr": lr * 0.1},  # Lower LR for pretrained CLIP vision
		{"params": model.image_encoder.projection.parameters(), "lr": lr},          # Normal LR for projection  
		{"params": [model.logit_scale], "lr": lr * 5},                              # Higher LR for temperature
	], weight_decay=0.01)  # Reduced weight decay
	# Simple step scheduler - more predictable
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
	scaler = GradScaler('cuda')  # For mixed precision
	if data_root is None:
		data_root = "../coco2014"  # Default path from Training_ folder
	train_loader, val_loader = create_dataloaders(data_root=data_root, batch_size=batch_size, max_samples=max_samples)
	train_losses, val_losses = [], []
	recall_history = {k: [] for k in ['Recall@1_i2t','Recall@1_t2i','Recall@5_i2t','Recall@5_t2i','Recall@10_i2t','Recall@10_t2i']}
	start_time = time.time()
	checkpoint_path = "clip_best_checkpoint.pth"
	start_epoch = 0
	epochs_run = 0
	# Resume from checkpoint if exists
	if os.path.exists(checkpoint_path):
		print(f"Resuming from checkpoint: {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
			train_losses = checkpoint['train_losses']
			val_losses = checkpoint['val_losses']
		if 'recall_history' in checkpoint:
			recall_history = checkpoint['recall_history']
		start_epoch = checkpoint.get('epoch', 1)
		epochs_run = checkpoint.get('epochs_run', start_epoch)
		print(f"Resuming from epoch {start_epoch}")
	else:
		print("Starting training from scratch.")

	for epoch in range(start_epoch, start_epoch + epochs):
		model.train()
		running_loss = 0.0
		train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
		for batch in train_iter:
			images = batch['image'].to(device)
			text_embeddings = batch['text_embedding'].to(device)
			optimizer.zero_grad()
			with autocast('cuda'):
				# Get image embeddings and use pre-computed text embeddings
				image_embeds, logit_scale = model(images, text_embeddings)
				# Both embeddings should already be normalized
				logits = logit_scale.exp() * torch.matmul(image_embeds, text_embeddings.t())
				
				# Hard negative mining - focus on difficult examples
				with torch.no_grad():
					diag_mask = torch.eye(logits.size(0), device=logits.device).bool()
					non_diag_logits = logits.masked_fill(diag_mask, float('-inf'))
					hard_negatives = non_diag_logits.max(dim=1)[0]
					pos_logits = logits.diag()
					margin = (hard_negatives - pos_logits).mean()
				
				loss = clip_infonce_loss(logits)
				# Add hard negative regularization
				if margin > 0:
					loss = loss + 0.1 * margin
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			
			# Update EMA model
			with torch.no_grad():
				for ema_param, param in zip(ema_model.parameters(), model.parameters()):
					ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
			
			running_loss += loss.item() * images.size(0)
			train_iter.set_postfix(loss=loss.item())
		avg_train_loss = running_loss / len(train_loader.dataset)
		train_losses.append(avg_train_loss)

		# Validation
		ema_model.eval()  # Use EMA model for validation
		val_loss = 0.0
		all_image_embeds = []
		all_text_embeds = []
		val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
		with torch.no_grad():
			for batch in val_iter:
				images = batch['image'].to(device)
				text_embeddings = batch['text_embedding'].to(device)
				with autocast('cuda'):
					# Get image embeddings and use pre-computed text embeddings
					image_embeds, logit_scale = ema_model(images, text_embeddings)
					# Both embeddings should already be normalized
					logits = logit_scale.exp() * torch.matmul(image_embeds, text_embeddings.t())
					loss = clip_infonce_loss(logits)
				val_loss += loss.item() * images.size(0)
				all_image_embeds.append(image_embeds.cpu())
				all_text_embeds.append(text_embeddings.cpu())
				val_iter.set_postfix(loss=loss.item())
		avg_val_loss = val_loss / len(val_loader.dataset)
		val_losses.append(avg_val_loss)

		# Compute Recall@K
		all_image_embeds = torch.cat(all_image_embeds, dim=0)
		all_text_embeds = torch.cat(all_text_embeds, dim=0)
		sim_matrix = compute_cosine_similarity_matrix(all_image_embeds, all_text_embeds)
		for k in [1, 5, 10]:
			r_i2t, r_t2i = recall_at_k(sim_matrix, k)
			recall_history[f'Recall@{k}_i2t'].append(r_i2t)
			recall_history[f'Recall@{k}_t2i'].append(r_t2i)
		scheduler.step()

		print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Temp: {model.logit_scale.exp().item():.2f}")
		print("  Recall@1_i2t: {:.4f}  Recall@1_t2i: {:.4f}  Recall@5_i2t: {:.4f}  Recall@5_t2i: {:.4f}  Recall@10_i2t: {:.4f}  Recall@10_t2i: {:.4f}".format(
			recall_history['Recall@1_i2t'][-1], recall_history['Recall@1_t2i'][-1],
			recall_history['Recall@5_i2t'][-1], recall_history['Recall@5_t2i'][-1],
			recall_history['Recall@10_i2t'][-1], recall_history['Recall@10_t2i'][-1]))

		# Save only the best checkpoint (lowest val loss)
		if epoch == start_epoch or avg_val_loss < min(val_losses[:-1]):
			# Remove previous checkpoint if exists
			if os.path.exists(checkpoint_path):
				os.remove(checkpoint_path)
			torch.save({
				'epoch': epoch + 1,
				'epochs_run': epochs_run + (epoch - start_epoch + 1),
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'train_losses': train_losses,
				'val_losses': val_losses,
				'recall_history': recall_history
			}, checkpoint_path)
			print(f"Best checkpoint saved: {checkpoint_path}")

		# Save plots every 3 epochs (and at last epoch)
		if ((epoch + 1) % 3 == 0) or (epoch == start_epoch + epochs - 1):
			plot_losses(train_losses, val_losses)
			plot_recalls(recall_history)
			print(f"Saved loss and recall plots at epoch {epoch+1}")
	total_time = time.time() - start_time
	return train_losses, val_losses, recall_history, total_time, device, model

# --------- Plotting and Reporting ---------
def plot_losses(train_losses, val_losses):
	plt.figure()
	plt.plot(train_losses, label='Train Loss')
	plt.plot(val_losses, label='Val Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('CLIP Training/Validation Loss Curves')
	output_dir = "standard_image"
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, 'clip_loss_curves.png'))
	plt.close()

def plot_recalls(recall_history):
	plt.figure(figsize=(10,6))
	for k in [1,5,10]:
		plt.plot(recall_history[f'Recall@{k}_i2t'], label=f'Recall@{k} i2t')
		plt.plot(recall_history[f'Recall@{k}_t2i'], label=f'Recall@{k} t2i', linestyle='--')
	plt.xlabel('Epoch')
	plt.ylabel('Recall')
	plt.legend()
	plt.title('Recall@K over Epochs')
	output_dir = "standard_image"
	os.makedirs(output_dir, exist_ok=True)
	plt.savefig(os.path.join(output_dir, 'clip_recall_curves.png'))
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
	import argparse
	parser = argparse.ArgumentParser(description="CLIP Training Script")
	parser.add_argument('--max_samples', type=int, default=100, help='Number of samples to use from the dataset')
	parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
	args = parser.parse_args()
	print(f"Starting CLIP training with max_samples={args.max_samples}, epochs={args.epochs}, batch_size={args.batch_size}...")
	# Set the correct path to coco2014 (relative to this script)
	coco2014_path = "../coco2014"  # Go up one directory from Training_ folder
	train_losses, val_losses, recall_history, total_time, device, model = train_clip(
		epochs=args.epochs, batch_size=args.batch_size, lr=3e-4, max_samples=args.max_samples, data_root=coco2014_path
	)
	plot_losses(train_losses, val_losses)
	plot_recalls(recall_history)
	report(total_time, device)
	print("Loss and recall curves saved as clip_loss_curves.png and clip_recall_curves.png")
