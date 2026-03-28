"""
Cardiotect - CalciumNet Model (V2)

U-Net with pretrained encoder (ConvNeXt/ResNet) and dual heads:
- Calcium segmentation (binary)
- Vessel classification (5-class with CoordConv for spatial awareness)

V2 Changes:
- GroupNorm instead of BatchNorm (batch-size independent)
- CoordConv on vessel head (explicit XY coordinates for spatial awareness)
- Stem skip connection for up4 (preserves high-resolution features)
- Deep supervision (auxiliary calcium heads at intermediate decoder stages)
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import timm  # type: ignore
import logging

from .config import USE_DEEP_SUPERVISION  # type: ignore

logger = logging.getLogger(__name__)


class GroupNormDoubleConv(nn.Module):
    """Double convolution block with GroupNorm (batch-size independent)."""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        # Ensure groups divides out_ch
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsampling + skip connection + double conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = GroupNormDoubleConv(in_ch // 2 + out_ch, out_ch)
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UpNoSkip(nn.Module):
    """Upsampling without skip connection (for final stage or when skip is optional)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = GroupNormDoubleConv(out_ch, out_ch)
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class StemBlock(nn.Module):
    """Lightweight stem for extracting stride-2 features (skip connection for up4)."""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.block(x)


class CoordConv(nn.Module):
    """Adds normalized XY coordinate channels to feature maps.
    
    This gives the vessel classification head explicit spatial awareness,
    allowing it to distinguish vessel territories by location in the image.
    """
    def __init__(self):
        super().__init__()
        self._coords_cached = None
        self._cached_size = None
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Cache coordinate grids for efficiency
        if self._cached_size != (h, w) or self._coords_cached is None or self._coords_cached.device != x.device:
            y_coords = torch.linspace(-1, 1, h, device=x.device)
            x_coords = torch.linspace(-1, 1, w, device=x.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            self._coords_cached = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, H, W)
            self._cached_size = (h, w)
        
        coords = self._coords_cached.expand(b, -1, -1, -1)  # (B, 2, H, W)
        return torch.cat([x, coords], dim=1)  # (B, C+2, H, W)


class CalciumNet(nn.Module):
    """U-Net with pretrained encoder and dual output heads.
    
    Architecture:
        Encoder: timm ConvNeXt Tiny (pretrained, 4-5 stages)
        Stem:    Conv 7×7 stride-2 → 64ch (provides skip for up4)
        Decoder: 4 upsampling blocks with skip connections + GroupNorm
        Heads:   - Calcium: 1-ch binary segmentation
                 - Vessel:  5-ch classification with CoordConv
        
    Deep Supervision:
        Auxiliary calcium heads at up2 and up3 outputs (0.3× weight).
    """
    def __init__(self, encoder_name='convnext_tiny', use_deep_supervision=None):
        super().__init__()
        
        if use_deep_supervision is None:
            use_deep_supervision = USE_DEEP_SUPERVISION
        self.use_deep_supervision = use_deep_supervision
        
        # --- Stem (for high-res skip connection) ---
        self.stem = StemBlock(in_ch=3, out_ch=64)
        
        # --- Encoder ---
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, 
            features_only=True, in_chans=3
        )
        
        # Detect encoder feature channels dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.encoder(dummy)
            ch = [f.shape[1] for f in features]
            logger.info(f"Encoder stages: {len(ch)}, channels: {ch}")
        
        # Use last 4 stages for decoder (handle 4 or 5 stage encoders)
        if len(ch) >= 5:
            self.encoder_stages = 5
            ch = ch[-4:]  # Use last 4 for decoder skips
            self._skip_offset = len(features) - 4
        else:
            self.encoder_stages = len(ch)
            self._skip_offset = 0
        
        # --- Decoder ---
        # up1: deepest features → ch[2] skip
        self.up1 = Up(ch[3], ch[2])
        # up2: ch[2] → ch[1] skip
        self.up2 = Up(ch[2], ch[1])
        # up3: ch[1] → ch[0] skip
        self.up3 = Up(ch[1], ch[0])
        # up4: ch[0] → stem skip (64ch)
        self.up4 = Up(ch[0], 64)
        # up5: final upsample to full resolution (no skip)
        self.up5 = UpNoSkip(64, 64)
        
        # --- Calcium Head ---
        self.out_calc = nn.Conv2d(64, 1, kernel_size=1)
        
        # --- Vessel Head (with CoordConv for spatial awareness) ---
        self.coord_conv = CoordConv()
        self.out_vessel = nn.Sequential(
            nn.Conv2d(64 + 2, 32, kernel_size=3, padding=1, bias=False),  # +2 for XY coords
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 5, kernel_size=1)  # 5 classes: bg + 4 vessels
        )
        
        # --- Deep Supervision Heads ---
        if self.use_deep_supervision:
            self.aux_head_up2 = nn.Conv2d(ch[1], 1, kernel_size=1)
            self.aux_head_up3 = nn.Conv2d(ch[0], 1, kernel_size=1)
        
        logger.info(f"CalciumNet V2 initialized: encoder={encoder_name}, "
                     f"deep_supervision={use_deep_supervision}, "
                     f"decoder_channels={ch}")

    def forward(self, x):
        # --- Stem features (stride-2, for up4 skip) ---
        stem_feat = self.stem(x)  # (B, 64, H/2, W/2)
        
        # --- Encoder ---
        features = self.encoder(x)
        
        # Get the 4 decoder skip features
        f0 = features[self._skip_offset]      # stride 4
        f1 = features[self._skip_offset + 1]  # stride 8
        f2 = features[self._skip_offset + 2]  # stride 16
        f3 = features[self._skip_offset + 3]  # stride 32 (deepest)
        
        # --- Decoder ---
        d1 = self.up1(f3, f2)    # stride 16
        d2 = self.up2(d1, f1)    # stride 8
        d3 = self.up3(d2, f0)    # stride 4
        d4 = self.up4(d3, stem_feat)  # stride 2 (uses stem skip!)
        d5 = self.up5(d4)        # stride 1 (full resolution)
        
        # --- Heads ---
        calc_logits = self.out_calc(d5)  # (B, 1, H, W)
        
        vessel_features = self.coord_conv(d5)  # (B, 66, H, W)
        vessel_logits = self.out_vessel(vessel_features)  # (B, 5, H, W)
        
        result = {
            'calc_logits': calc_logits,
            'vessel_logits': vessel_logits,
        }
        
        # --- Deep Supervision ---
        if self.use_deep_supervision and self.training:
            aux_logits = [
                self.aux_head_up2(d2),  # stride 8
                self.aux_head_up3(d3),  # stride 4
            ]
            result['aux_calc_logits'] = aux_logits
        
        return result
