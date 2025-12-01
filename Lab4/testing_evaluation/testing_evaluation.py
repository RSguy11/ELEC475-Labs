# CLIP Evaluation and Visualization for 2.4

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CLIP_model_design.clip_pipe import CLIPModel
from Load_data_set.load_data_set import create_dataloaders


def compute_cosine_similarity_matrix(image_embeds, text_embeds):
	# image_embeds: [N, D], text_embeds: [N, D]
	image_embeds = F.normalize(image_embeds, p=2, dim=1)
	text_embeds = F.normalize(text_embeds, p=2, dim=1)
	return image_embeds @ text_embeds.T

def recall_at_k(sim_matrix, k=1):
	# sim_matrix: [N, N], ground truth is diagonal
	N = sim_matrix.size(0)
	# Image-to-Text
	i2t_ranks = sim_matrix.argsort(dim=1, descending=True)
	i2t_hits = [(i in i2t_ranks[i, :k]) for i in range(N)]
	recall_i2t = np.mean(i2t_hits)
	# Text-to-Image
	t2i_ranks = sim_matrix.argsort(dim=0, descending=True)
	t2i_hits = [(i in t2i_ranks[:k, i]) for i in range(N)]
	recall_t2i = np.mean(t2i_hits)
	return recall_i2t, recall_t2i

def evaluate_clip(model, dataloader, device):
	model.eval()
	all_image_embeds = []
	all_text_embeds = []
	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Evaluating", leave=False):
			images = batch['image'].to(device)
			texts = batch['text']
			image_embeds = model.encode_image(images).cpu()
			text_embeds = model.encode_text(texts).cpu()
			all_image_embeds.append(image_embeds)
			all_text_embeds.append(text_embeds)
	all_image_embeds = torch.cat(all_image_embeds, dim=0)
	all_text_embeds = torch.cat(all_text_embeds, dim=0)
	sim_matrix = compute_cosine_similarity_matrix(all_image_embeds, all_text_embeds)
	recalls = {}
	for k in [1, 5, 10]:
		r_i2t, r_t2i = recall_at_k(sim_matrix, k)
		recalls[f'Recall@{k}_i2t'] = r_i2t
		recalls[f'Recall@{k}_t2i'] = r_t2i
	return sim_matrix, recalls, all_image_embeds, all_text_embeds

def visualize_text_to_image(model, dataloader, device, query, top_k=5):
	model.eval()
	# Get all image embeddings
	images_list = []
	image_embeds_list = []
	with torch.no_grad():
		for batch in dataloader:
			images = batch['image'].to(device)
			image_embeds = model.encode_image(images).cpu()
			# batch['image_path'] is a list of paths
			image_paths = batch['image_path'] if 'image_path' in batch else [None]*images.size(0)
			images_list.extend(image_paths)
			image_embeds_list.append(image_embeds)
	all_image_embeds = torch.cat(image_embeds_list, dim=0)
	# Encode query
	text_embed = model.encode_text([query]).cpu()
	sims = F.normalize(all_image_embeds, p=2, dim=1) @ F.normalize(text_embed, p=2, dim=1).T
	topk_idx = sims.squeeze().argsort(descending=True)[:top_k]
	print(f"Top-{top_k} images for query '{query}':")
	output_dir = "standard_image"
	os.makedirs(output_dir, exist_ok=True)
	for rank, idx in enumerate(topk_idx, 1):
		path = images_list[idx] if images_list[idx] else f"Index {idx}"
		print(f"  {path}")
		try:
			img = Image.open(path)
			plt.imshow(img)
			plt.title(f"Text-to-Image Retrieval\nQuery: '{query}' | Top-{top_k} | Rank {rank}")
			plt.axis('off')
			out_path = os.path.join(output_dir, f"text2img_query_{query.replace(' ','_')}_rank{rank}.png")
			plt.savefig(out_path)
			plt.close()
			print(f"Saved: {out_path}")
		except Exception as e:
			print(f"Could not display/save image: {e}")

def visualize_best_and_worst(model, dataloader, device, query, top_k=5):
	"""Visualize top-k best and worst retrievals for a category query."""
	model.eval()
	images_list = []
	image_embeds_list = []
	with torch.no_grad():
		for batch in dataloader:
			images = batch['image'].to(device)
			image_embeds = model.encode_image(images).cpu()
			image_paths = batch['image_path'] if 'image_path' in batch else [None]*images.size(0)
			images_list.extend(image_paths)
			image_embeds_list.append(image_embeds)
	all_image_embeds = torch.cat(image_embeds_list, dim=0)
	text_embed = model.encode_text([query]).cpu()
	sims = F.normalize(all_image_embeds, p=2, dim=1) @ F.normalize(text_embed, p=2, dim=1).T
	sims = sims.squeeze()
	topk_idx = sims.argsort(descending=True)[:top_k]
	bottomk_idx = sims.argsort(descending=False)[:top_k]
	output_dir = "standard_image"
	os.makedirs(output_dir, exist_ok=True)
	# Best
	for rank, idx in enumerate(topk_idx, 1):
		path = images_list[idx] if images_list[idx] else f"Index {idx}"
		try:
			img = Image.open(path)
			plt.imshow(img)
			plt.title(f"Best Retrieval for Query: '{query}'\nTop-{top_k} | Rank {rank}")
			plt.axis('off')
			out_path = os.path.join(output_dir, f"best_{query.replace(' ','_')}_rank{rank}.png")
			plt.savefig(out_path)
			plt.close()
		except Exception as e:
			print(f"Could not display/save image: {e}")
	# Worst
	for rank, idx in enumerate(bottomk_idx, 1):
		path = images_list[idx] if images_list[idx] else f"Index {idx}"
		try:
			img = Image.open(path)
			plt.imshow(img)
			plt.title(f"Worst Retrieval for Query: '{query}'\nLowest-{top_k} | Rank {rank}")
			plt.axis('off')
			out_path = os.path.join(output_dir, f"worst_{query.replace(' ','_')}_rank{rank}.png")
			plt.savefig(out_path)
			plt.close()
		except Exception as e:
			print(f"Could not display/save image: {e}")

def visualize_image_best_guesses(model, dataloader, device, image_idx=0, class_list=None, top_k=5):
	"""Visualize the model's top-k best guesses for a given image over a class list."""
	model.eval()
	# Get all images and paths
	images = []
	image_paths = []
	for batch in dataloader:
		for img, path in zip(batch['image'], batch['image_path']):
			images.append(img)
			image_paths.append(path)
	if image_idx >= len(images):
		print(f"Image index {image_idx} out of range.")
		return
	img_tensor = images[image_idx].unsqueeze(0).to(device)
	img_path = image_paths[image_idx]
	with torch.no_grad():
		text_embeds = model.encode_text(class_list).cpu()
		image_embed = model.encode_image(img_tensor).cpu()
		sims = F.normalize(image_embed, p=2, dim=1) @ F.normalize(text_embeds, p=2, dim=1).T
		sims = sims.squeeze()
		topk_idx = sims.argsort(descending=True)[:top_k]
	output_dir = "standard_image"
	os.makedirs(output_dir, exist_ok=True)
	img = Image.open(img_path)
	for rank, idx in enumerate(topk_idx, 1):
		plt.imshow(img)
		plt.title(f"Image-to-Text Classification\nImage: {os.path.basename(img_path)} | Guess {rank}: {class_list[idx]}")
		plt.axis('off')
		out_path = os.path.join(output_dir, f"img_best_guess_{os.path.basename(img_path)}_guess{rank}_{class_list[idx].replace(' ','_')}.png")
		plt.savefig(out_path)
		plt.close()

def classify_image(model, image_path, device, class_list):
	model.eval()
	from torchvision import transforms
	CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
	CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
	preprocess = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
	])
	img = Image.open(image_path).convert('RGB')
	img_tensor = preprocess(img).unsqueeze(0).to(device)
	with torch.no_grad():
		image_embed = model.encode_image(img_tensor).cpu()
		text_embeds = model.encode_text(class_list).cpu()
		sims = F.normalize(image_embed, p=2, dim=1) @ F.normalize(text_embeds, p=2, dim=1).T
		top_idx = sims.squeeze().argmax().item()
		print(f"Image classified as: {class_list[top_idx]}")
	output_dir = "standard_training_results"
	os.makedirs(output_dir, exist_ok=True)
	plt.imshow(img)
	plt.title(f"Predicted: {class_list[top_idx]}")
	plt.axis('off')
	out_path = os.path.join(output_dir, f"img_classify_{os.path.basename(image_path)}_pred_{class_list[top_idx].replace(' ','_')}.png")
	plt.savefig(out_path)
	plt.close()
	print(f"Saved: {out_path}")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="CLIP Evaluation Script")
	parser.add_argument('--max_samples', type=int, default=100, help='Number of samples to use from the dataset')
	args = parser.parse_args()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Load model and data
	model = CLIPModel().to(device)
	# Optionally, load a checkpoint here
	# model.load_state_dict(torch.load('clip_checkpoint_epoch5.pth')['model_state_dict'])
	_, val_loader = create_dataloaders(batch_size=16, max_samples=args.max_samples)

	# Evaluation
	sim_matrix, recalls, image_embeds, text_embeds = evaluate_clip(model, val_loader, device)
	print("Recall metrics:")
	for k, v in recalls.items():
		print(f"  {k}: {v:.4f}")


	# Visualize best and worst for a category
	visualize_best_and_worst(model, val_loader, device, query='sport', top_k=5)

	# Visualize best guesses for a single image
	class_list = ['a person', 'an animal', 'a landscape', 'a car', 'a building']
	visualize_image_best_guesses(model, val_loader, device, image_idx=0, class_list=class_list, top_k=5)

	print("Discuss: Analyze recall metrics and training/validation loss curves to discuss performance trends and training dynamics.")
