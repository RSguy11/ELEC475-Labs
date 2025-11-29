# CLIP Evaluation and Visualization for 2.4

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from CLIP_model_design.clip_pipe import CLIPModel
from Lab4.Load_data_set.load_data_set import create_dataloaders

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
		for batch in dataloader:
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
			image_paths = batch.get('image_path', [None]*images.size(0))
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
			plt.title(f"Rank {rank}")
			plt.axis('off')
			out_path = os.path.join(output_dir, f"text2img_query_{query.replace(' ','_')}_rank{rank}.png")
			plt.savefig(out_path)
			plt.close()
			print(f"Saved: {out_path}")
		except Exception as e:
			print(f"Could not display/save image: {e}")

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
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Load model and data
	model = CLIPModel().to(device)
	# Optionally, load a checkpoint here
	# model.load_state_dict(torch.load('clip_checkpoint_epoch5.pth')['model_state_dict'])
	_, val_loader = create_dataloaders(batch_size=16, max_samples=100)

	# Evaluation
	sim_matrix, recalls, image_embeds, text_embeds = evaluate_clip(model, val_loader, device)
	print("Recall metrics:")
	for k, v in recalls.items():
		print(f"  {k}: {v:.4f}")

	# Visualize text-to-image retrieval
	visualize_text_to_image(model, val_loader, device, query='sport', top_k=5)

	# Visualize image classification
	# Provide a real image path and class list for your dataset
	# classify_image(model, 'path/to/image.jpg', device, ['a person', 'an animal', 'a landscape'])

	print("Discuss: Analyze recall metrics and training/validation loss curves to discuss performance trends and training dynamics.")
