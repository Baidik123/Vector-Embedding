import {
  Component,
  Input,
  signal,
  OnChanges,
  SimpleChanges,
  Output,
  EventEmitter,
} from '@angular/core';
import { CommonModule } from '@angular/common';

export type AlgorithmId = 'dhash' | 'dino_small' | 'dino_large' | 'dino_giant' | 'dino_base' | 'clip' | 'efficientnet' | 'convnext' | 'resnet' | 'hybrid';

interface AlgorithmStatus {
  id: AlgorithmId;
  label: string;
  loading: boolean;
  error: string;
  similarity: number | null;
  band: string | null;
  bandDescription: string | null;
  hammingDistance?: number;
  dhashSimilarity?: number;       // hybrid only
  dinoSimilarity?: number;        // hybrid only
  dinoLargeSimilarity?: number;   // hybrid only
}

const ALGORITHM_LABELS: Record<AlgorithmId, string> = {
  dhash:      'dHash',
  dino_small: 'DINO V2 Small',
  dino_large: 'DINO V2 Large',
  dino_giant: 'DINO V2 Giant',
  dino_base:    'DINO V2 Base',
  clip:         'CLIP',
  efficientnet: 'EfficientNet-B4',
  convnext:     'ConvNeXt-Tiny',
  resnet:       'ResNet-50',
  hybrid:       'Hybrid',
};

const ALGORITHM_ENDPOINTS: Record<AlgorithmId, string> = {
  dhash:      '/compare/dhash',
  dino_small: '/compare/dino',
  dino_large: '/compare/dino-large',
  dino_giant: '/compare/dino-giant',
  dino_base:    '/compare/dino-base',
  clip:         '/compare/clip',
  efficientnet: '/compare/efficientnet',
  convnext:     '/compare/convnext',
  resnet:       '/compare/resnet',
  hybrid:       '/compare/hybrid',
};

@Component({
  selector: 'app-pdf-compare',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './pdf-compare.html',
  styleUrl: './pdf-compare.css',
})
export class PdfCompare implements OnChanges {
  @Input() firstPngUrl: string | null = null;
  @Input() secondPngUrl: string | null = null;
  @Input() firstFile: File | null = null;
  @Input() secondFile: File | null = null;
  @Input() selectedAlgorithms: AlgorithmId[] = [];
  @Output() closeCompare = new EventEmitter<void>();

  sliderValue = signal(50);
  zoomLevel = signal(1.0);
  algorithmStatuses = signal<AlgorithmStatus[]>([]);

  private readonly API_URL = 'http://localhost:8000';

  ngOnChanges(changes: SimpleChanges): void {
    const relevant =
      changes['firstPngUrl'] ||
      changes['secondPngUrl'] ||
      changes['selectedAlgorithms'];
    if (relevant && this.firstFile && this.secondFile && this.selectedAlgorithms.length > 0) {
      this.sliderValue.set(50);
      this.zoomLevel.set(1.0);
      this.performComparison();
    }
  }

  private performComparison(): void {
    if (!this.firstFile || !this.secondFile) return;

    this.algorithmStatuses.set(
      this.selectedAlgorithms.map((id) => ({
        id,
        label: ALGORITHM_LABELS[id],
        loading: true,
        error: '',
        similarity: null,
        band: null,
        bandDescription: null,
      }))
    );

    for (const id of this.selectedAlgorithms) {
      this.runAlgorithm(id);
    }
  }

  private runAlgorithm(id: AlgorithmId): void {
    const formData = new FormData();
    formData.append('first', this.firstFile!);
    formData.append('second', this.secondFile!);

    fetch(`${this.API_URL}${ALGORITHM_ENDPOINTS[id]}`, { method: 'POST', body: formData })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((body) => {
            throw new Error(body?.detail ?? `HTTP ${res.status}`);
          });
        }
        return res.json();
      })
      .then((data) => {
        this.algorithmStatuses.update((statuses) =>
          statuses.map((s) =>
            s.id === id
              ? {
                  ...s,
                  loading: false,
                  similarity: data.similarity,
                  band: data.band,
                  bandDescription: data.band_description,
                  hammingDistance: data.hamming_distance,
                  dhashSimilarity: data.dhash_similarity,
                  dinoSimilarity: data.dino_similarity,
                  dinoLargeSimilarity: data.dino_large_similarity,
                }
              : s
          )
        );
      })
      .catch((err: Error) => {
        this.algorithmStatuses.update((statuses) =>
          statuses.map((s) =>
            s.id === id
              ? { ...s, loading: false, error: err.message || 'Comparison failed' }
              : s
          )
        );
      });
  }

  isAnyLoading(): boolean {
    return this.algorithmStatuses().some((s) => s.loading);
  }

  onSliderChange(event: Event): void {
    this.sliderValue.set(Number((event.target as HTMLInputElement).value));
  }

  zoomIn(): void {
    this.zoomLevel.update((v) => Math.min(v + 0.1, 3.0));
  }

  zoomOut(): void {
    this.zoomLevel.update((v) => Math.max(v - 0.1, 0.5));
  }

  resetZoom(): void {
    this.zoomLevel.set(1.0);
  }

  getBandColor(band: string | null): string {
    switch (band) {
      case 'Exact Duplicate':      return '#ef4444';
      case 'Likely Duplicate':     return '#f97316';
      case 'Similar – Same Family': return '#22c55e';
      case 'Similar – Related':    return '#3b82f6';
      case 'Different':            return '#9ca3af';
      default:                     return '#9ca3af';
    }
  }

  absScore(v: number): number {
    return Math.abs(v);
  }

  onClose(): void {
    this.closeCompare.emit();
  }
}
