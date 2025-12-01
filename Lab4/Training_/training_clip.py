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
def clip_infonce_loss(logits):
	# logits: [batch_size, batch_size] similarity matrix
	labels = torch.arange(logits.size(0)).to(logits.device)
	loss_i2t = nn.CrossEntropyLoss()(logits, labels)
	loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
	return (loss_i2t + loss_t2i) / 2

# --------- Training Function ---------

def compute_cosine_similarity_matrix(image_embeds, text_embeds):
	image_embeds = F.normalize(image_embeds, p=2, dim=1)
	text_embeds = F.normalize(text_embeds, p=2, dim=1)
	return image_embeds @ text_embeds.T

def recall_at_k(sim_matrix, k=1):
	N = sim_matrix.size(0)
	i2t_ranks = sim_matrix.argsort(dim=1, descending=True)
	i2t_hits = [(i in i2t_ranks[i, :k]) for i in range(N)]
	recall_i2t = np.mean(i2t_hits)
	t2i_ranks = sim_matrix.argsort(dim=0, descending=True)
	t2i_hits = [(i in t2i_ranks[:k, i]) for i in range(N)]
	recall_t2i = np.mean(t2i_hits)
	return recall_i2t, recall_t2i

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
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
	scaler = GradScaler('cuda')  # For mixed precision
	if data_root is None:
		data_root = "coco2014"
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
		checkpoint = torch.load(checkpoint_path, map_location=device)
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
			texts = batch['text']  # list of strings
			optimizer.zero_grad()
			with autocast('cuda'):
				out = model(images, texts)
				loss = clip_infonce_loss(out['logits'])
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			running_loss += loss.item() * images.size(0)
			train_iter.set_postfix(loss=loss.item())
		avg_train_loss = running_loss / len(train_loader.dataset)
		train_losses.append(avg_train_loss)

		# Validation
		model.eval()
		val_loss = 0.0
		all_image_embeds = []
		all_text_embeds = []
		val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
		with torch.no_grad():
			for batch in val_iter:
				images = batch['image'].to(device)
				texts = batch['text']
				with autocast('cuda'):
					out = model(images, texts)
					loss = clip_infonce_loss(out['logits'])
				val_loss += loss.item() * images.size(0)
				all_image_embeds.append(model.encode_image(images).cpu())
				all_text_embeds.append(model.encode_text(texts).cpu())
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
		scheduler.step(avg_val_loss)

		print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
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
	# Set the correct path to coco2014 (relative to this script)
	coco2014_path = "coco2014"  # adjust if needed
	train_losses, val_losses, recall_history, total_time, device, model = train_clip(
		epochs=args.epochs, batch_size=args.batch_size, lr=1e-4, max_samples=args.max_samples, data_root=coco2014_path
	)
	plot_losses(train_losses, val_losses)
	plot_recalls(recall_history)
	report(total_time, device)
	print("Loss and recall curves saved as clip_loss_curves.png and clip_recall_curves.png")
