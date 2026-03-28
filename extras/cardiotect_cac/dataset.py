import os
import json
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
import numpy as np # type: ignore
import logging
from pathlib import Path
from tqdm import tqdm # type: ignore

from .config import ( # type: ignore
    DATASET_ROOT, PATIENT_SUBPATH_TEMPLATE, CALCIUM_XML_PATH,
    IMAGE_SIZE, GATED_REL_PATH, HU_CLIP_MIN, HU_CLIP_MAX,
    AGATSTON_MIN_HU
)
from .dicom_io import load_dicom_series, preprocess_volume # type: ignore
from .xml_io import parse_calcium_xml, align_xml_to_dicom, create_mask_from_rois # type: ignore
import pydicom # type: ignore
import albumentations as A # type: ignore

logger = logging.getLogger(__name__)

class CardiotectDataset(Dataset):
    """
    Dataset for Calcium Scoring.
    Manages loading of processed volumes and masks.
    Supports dynamic sample updates for hard negative mining.
    Now supports CAC-0 patients (no XML annotations) as pure negatives.
    """
    def __init__(self, root_dir=DATASET_ROOT, subset="train", cache_ram=False):
        """
        Args:
            root_dir (Path): Root dataset path.
            subset (str): 'train' or 'val'.
            cache_ram (bool): Ignored. Forced to False to prevent OOM.
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        
        # 1. Scan for patients
        base_path = self.root_dir / GATED_REL_PATH / "patient"
        xml_path = self.root_dir / CALCIUM_XML_PATH
        if not base_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {base_path}")
            
        all_patient_ids = [d for d in os.listdir(base_path) if os.path.isdir(base_path / d)]
        import random
        all_patient_ids.sort() # Sort first to ensure pure determinism across OS
        
        # 2. Identify CAC-0 patients (those WITHOUT XML annotations)
        xml_files = set(f.replace('.xml', '') for f in os.listdir(xml_path) if f.endswith('.xml')) if xml_path.exists() else set()
        self.annotated_patient_ids = [pid for pid in all_patient_ids if pid in xml_files]
        self.cac0_patient_ids = [pid for pid in all_patient_ids if pid not in xml_files]
        
        logger.info(f"Found {len(self.annotated_patient_ids)} annotated patients, {len(self.cac0_patient_ids)} CAC-0 patients")
        
        # 3. Deterministic Shuffle for Fair 80/20 Split (Seed 42)
        rng = random.Random(42)
        rng.shuffle(self.annotated_patient_ids)
        rng.shuffle(self.cac0_patient_ids)
        
        split_idx_ann = int(0.8 * len(self.annotated_patient_ids))
        split_idx_cac0 = int(0.8 * len(self.cac0_patient_ids))
        
        if subset == "train":
            self.patient_ids = self.annotated_patient_ids[:split_idx_ann] # type: ignore
            self.cac0_patient_ids_subset = self.cac0_patient_ids[:split_idx_cac0] # type: ignore
        else:
            self.patient_ids = self.annotated_patient_ids[split_idx_ann:] # type: ignore
            self.cac0_patient_ids_subset = self.cac0_patient_ids[split_idx_cac0:] # type: ignore
            
        self.samples = []  # Annotated patient samples
        self.cac0_samples = []  # CAC-0 patient samples (pure negatives)
        
        self._prepare_data()
        
    def _prepare_data(self):
        """
        Scans patients, performs alignment check to map XML->DICOM, 
        and stores sample metadata. Does NOT keep volumes in RAM.
        Now with persistent JSON caching!
        """
        cache_file = self.root_dir / f"dataset_cache_{self.subset}.json"
        
        # Try loading cache
        # Try loading cache
        if cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}...")
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.samples = data.get('samples', [])
                    self.cac0_samples = data.get('cac0_samples', [])
                logger.info(f"Loaded {len(self.samples)} samples and {len(self.cac0_samples)} CAC-0 samples from cache.")
                
                # IMPORTANT: Must build lookup map even when loaded from cache!
                self._build_patient_map()
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-indexing.")
        
        logger.info(f"Preparing {self.subset} dataset ({len(self.patient_ids)} patients)...")
        
        if not hasattr(self, 'cac0_patient_ids_subset'):
             self.cac0_patient_ids_subset = []

        # Process Annotated Patients
        for pid in tqdm(self.patient_ids, desc="Indexing Annotated Data"):
            # Construct paths
            patient_folder = self.root_dir / GATED_REL_PATH / "patient" / pid / "Pro_Gated_CS_3.0_I30f_3_70%"
            xml_filename = f"{pid}.xml" 
            xml_path = self.root_dir / CALCIUM_XML_PATH / xml_filename
            
            if not patient_folder.exists():
                logger.debug(f"Missing folder for {pid}, skipping.")
                continue
            
            # 1. Load DICOM Metadata + Volume for Alignment (Temporary)
            try:
                # We need the full volume once to calculate the alignment offset
                slices, volume_hu, _ = load_dicom_series(str(patient_folder))
            except Exception as e:
                logger.warning(f"Failed load {pid}: {e}")
                continue
                
            # 2. Load XML
            if xml_path.exists():
                xml_data = parse_calcium_xml(str(xml_path))
            else:
                xml_data = {} 
            
            # 3. Compute Alignment (Offset)
            # This returns a map: {dicom_z_index: [rois]}
            aligned_rois_map = align_xml_to_dicom(xml_data, volume_hu, slices)
            
            # 4. Generate Samples
            num_slices = len(slices)
            for z in range(num_slices):
                dcm_file = slices[z].filename # type: ignore
                rois = aligned_rois_map.get(z, [])
                
                # Check for positive
                is_positive = (len(rois) > 0)
                
                self.samples.append({
                    'pid': pid,
                    'slice_idx': z,
                    'dcm_path': str(dcm_file),
                    'rois': rois,
                    'is_positive': is_positive
                })
            
            # 5. Cleanup to free RAM immediately
            del slices
            del volume_hu
            
        # Process CAC-0 (Pure Negative) Patients
        if self.cac0_patient_ids_subset:
            logger.info(f"Indexing {len(self.cac0_patient_ids_subset)} CAC-0 patients...")
            for pid in tqdm(self.cac0_patient_ids_subset, desc="Indexing CAC-0 Data"):
                patient_folder = self.root_dir / GATED_REL_PATH / "patient" / pid / "Pro_Gated_CS_3.0_I30f_3_70%"
                
                if not patient_folder.exists():
                    continue
                    
                # Just need file listing, no volume load needed (no alignment for negative patients)
                # But we need consistent sorting
                dcm_files = sorted(list(patient_folder.glob("*.dcm")))
                
                for z, dcm_file in enumerate(dcm_files):
                    self.cac0_samples.append({
                        'pid': pid,
                        'slice_idx': z,
                        'dcm_path': str(dcm_file),
                        'rois': [],
                        'is_positive': False
                    })

        logger.info(f"Indexed {len(self.samples)} annotated slices and {len(self.cac0_samples)} CAC-0 slices.")
        
        # Save cache
        try:
            cache_file = self.root_dir / f"dataset_cache_{self.subset}.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'samples': self.samples,
                    'cac0_samples': self.cac0_samples
                }, f)
            logger.info(f"Saved dataset cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        # 5. Build lookup map for 2.5D loading
        self._build_patient_map()

    def _build_patient_map(self):
        """Builds the pid -> slice_idx -> dcm_path map for 2.5D loading."""
        self.patient_map = {}
        
        # Helper for map building
        def _add_to_map(sample_list):
            for s in sample_list:
                pid = s['pid']
                z = s['slice_idx']
                if pid not in self.patient_map:
                    self.patient_map[pid] = {}
                self.patient_map[pid][z] = s['dcm_path']

        # Map active samples (or backup if available)
        samples_source = self.samples
        if hasattr(self, 'all_samples_backup'):
            samples_source = self.all_samples_backup
            
        _add_to_map(samples_source)
        
        # Also map CAC-0 samples so we can load their neighbors
        if hasattr(self, 'cac0_samples'):
            _add_to_map(self.cac0_samples)

    def set_active_samples(self, samples_list):
        self.samples = samples_list

    def set_mode_positive_only(self):
        """Sets active samples to ONLY positive slices (for initial training)."""
        # Ensure we back up the full dataset first!
        self.get_all_samples()
        
        self.samples = [s for s in self.samples if s['is_positive']]
        logger.info(f"Dataset set to POSITIVE ONLY mode. Count: {len(self.samples)}")

    def set_mode_balanced(self, neg_ratio=1.0):
        """
        Sets active samples to Positives + Random Negatives.
        neg_ratio: Number of negatives per positive.
        """
        self.get_all_samples()
        
        positives = [s for s in self.all_samples_backup if s['is_positive']]
        negatives = [s for s in self.all_samples_backup if not s['is_positive']]
        
        n_pos = len(positives)
        n_neg = int(n_pos * neg_ratio)
        
        # Sample Negatives
        import random
        selected_negatives = random.sample(negatives, min(len(negatives), n_neg))
        
        self.samples = positives + selected_negatives
        logger.info(f"Dataset set to BALANCED mode (Ratio {neg_ratio}). Pos: {n_pos}, Neg: {len(selected_negatives)}")

    def set_mode_with_pure_negatives(self, neg_ratio=1.0):
        """
        Mixes positive samples with Pure Negative (CAC-0) samples.
        This provides high-quality negatives that definitely have no calcium.
        """
        self.get_all_samples()
        
        # Get all positives
        positives = [s for s in self.all_samples_backup if s['is_positive']]
        
        # Get negatives: prioritize CAC-0 samples
        cac0_negatives = getattr(self, 'cac0_samples', [])
        
        # We also might want some negatives from the annotated set (hard negatives?)
        # For now, let's mix: 70% from pure negatives (CAC-0), 30% from annotated negatives
        annotated_negatives = [s for s in self.all_samples_backup if not s['is_positive']]
        
        n_pos = len(positives)
        n_total_neg = int(n_pos * neg_ratio)
        
        n_cac0 = int(n_total_neg * 0.7)
        n_ann = n_total_neg - n_cac0
        
        import random
        # Sample CAC-0
        selected_cac0 = random.sample(cac0_negatives, min(len(cac0_negatives), n_cac0))
        # Sample Annotated Negatives
        selected_ann = random.sample(annotated_negatives, min(len(annotated_negatives), n_ann))
        
        self.samples = positives + selected_cac0 + selected_ann
        logger.info(f"Dataset set to PURE NEGATIVE mode (Ratio {neg_ratio}). Pos: {n_pos}, CAC-0 Neg: {len(selected_cac0)}, Ann Neg: {len(selected_ann)}")

    def add_hard_negatives(self, negative_samples):
        """Adds a list of negative samples to the active training set."""
        if not negative_samples:
            return
        
        # Avoid duplicates (simple check by ID or just append if we trust caller)
        # Using a set of tuples for dedup might be slow, let's just append for speed 
        # assuming caller (Trainer) manages the pool.
        self.samples.extend(negative_samples)
        logger.info(f"Added {len(negative_samples)} hard negatives. New Count: {len(self.samples)}")
        
    def get_all_samples(self):
        """Returns the full master list of annotated AND CAC-0 samples."""
        if not hasattr(self, 'all_samples_backup'):
            self.all_samples_backup = list(self.samples)
        
        # Combine annotated sick patients and healthy CAC-0 patients for complete evaluation
        return self.all_samples_backup + getattr(self, 'cac0_samples', [])
    
    def reset_to_all(self):
        """Restores full dataset."""
        if hasattr(self, 'all_samples_backup'):
            self.samples = list(self.all_samples_backup)
        logger.info(f"Reset dataset to FULL mode. Count: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _get_augmentor(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.7
            ),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0, p=0.2
            ),
        ], additional_targets={'mask_vessel': 'mask'})

    def _load_slice_hu(self, dcm_path):
        """Load a single slice and return raw HU values."""
        try:
            dcm = pydicom.dcmread(dcm_path)
            raw = dcm.pixel_array.astype(np.float32)
            slope = getattr(dcm, 'RescaleSlope', 1.0)
            intercept = getattr(dcm, 'RescaleIntercept', 0.0)
            return raw * slope + intercept
        except Exception:
            return np.zeros((512, 512), dtype=np.float32)

    def _normalize_hu(self, slice_hu):
        """Clip and normalize HU values to [-1, 1]."""
        slice_clipped = np.clip(slice_hu, HU_CLIP_MIN, HU_CLIP_MAX)
        clip_range = HU_CLIP_MAX - HU_CLIP_MIN
        center = HU_CLIP_MIN + clip_range / 2.0
        return (slice_clipped - center) / (clip_range / 2.0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample['pid']
        z = sample['slice_idx']
        rois = sample['rois']
        
        # 1. Load Center Slice + Neighbors (2.5D)
        path_z = sample['dcm_path']
        path_zm1 = self.patient_map[pid].get(z - 1, path_z)
        path_zp1 = self.patient_map[pid].get(z + 1, path_z)
        
        # Load raw HU for center slice (needed for HU-aware mask filtering)
        hu_z = self._load_slice_hu(path_z)
        
        # Normalize all slices
        s_z = self._normalize_hu(hu_z)
        s_zm1 = self._normalize_hu(self._load_slice_hu(path_zm1))
        s_zp1 = self._normalize_hu(self._load_slice_hu(path_zp1))
        
        # Stack: (H, W, 3) for albumentations
        img_stack = np.stack([s_zm1, s_z, s_zp1], axis=-1)
        
        # 2. Create Masks (Only for Center Slice Z)
        c_mask, v_mask = create_mask_from_rois(rois, shape=(512, 512))
        
        # 3. HU-aware mask filtering: zero out mask where HU <= 130
        # This ensures training is consistent with the Agatston definition
        # and inference post-processing
        if c_mask.any():
            hu_filter = (hu_z > AGATSTON_MIN_HU).astype(np.uint8)
            c_mask = c_mask * hu_filter
            v_mask = v_mask * hu_filter
        
        # 4. Augmentation
        if self.subset == 'train':
            if not hasattr(self, 'augmentor'):
                self.augmentor = self._get_augmentor()
            
            img_stack = img_stack.astype(np.float32)
            augmented = self.augmentor(image=img_stack, mask=c_mask, mask_vessel=v_mask)
            img_stack = augmented['image']
            c_mask = augmented['mask']
            v_mask = augmented['mask_vessel']
        
        # 5. Format Output: (C, H, W)
        input_tensor = img_stack.transpose(2, 0, 1).astype(np.float32)
        c_mask = c_mask[np.newaxis, ...].astype(np.float32)  # (1, H, W)
        
        return {
            'image': torch.from_numpy(input_tensor),
            'mask_calc': torch.from_numpy(c_mask),
            'mask_vessel': torch.from_numpy(v_mask.astype(np.int64)),
            'pid': sample['pid'],
            'slice_idx': sample['slice_idx'],
            'is_positive': sample['is_positive']
        }
