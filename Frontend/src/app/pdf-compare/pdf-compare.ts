import { Component, Input, signal, OnChanges, SimpleChanges, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SafeResourceUrl } from '@angular/platform-browser';

interface ComparisonResult {
  hamming_distance: number;
  dhash_similarity: number;
  hybrid_score: number;
  band: string;
  band_description: string;
}

@Component({
  selector: 'app-pdf-compare',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './pdf-compare.html',
  styleUrl: './pdf-compare.css',
})
export class PdfCompare implements OnChanges {
  @Input() firstPngUrl: SafeResourceUrl | null = null;
  @Input() secondPngUrl: SafeResourceUrl | null = null;
  @Input() firstFile: File | null = null;
  @Input() secondFile: File | null = null;
  @Output() closeCompare = new EventEmitter<void>();

  sliderValue = signal(50);
  zoomLevel = signal(1.0);
  comparisonResult = signal<ComparisonResult | null>(null);
  isLoading = signal(false);
  errorMessage = signal('');

  private readonly API_URL = 'http://localhost:8000';

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['firstPngUrl'] || changes['secondPngUrl']) && this.firstFile && this.secondFile) {
      this.sliderValue.set(50);
      this.zoomLevel.set(1.0);
      this.performComparison();
    }
  }

  private performComparison(): void {
    if (!this.firstFile || !this.secondFile) {
      this.errorMessage.set('Both files are required for comparison.');
      return;
    }

    this.isLoading.set(true);
    this.errorMessage.set('');
    this.comparisonResult.set(null);

    const formData = new FormData();
    formData.append('first', this.firstFile);
    formData.append('second', this.secondFile);

    fetch(`${this.API_URL}/compare`, {
      method: 'POST',
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data: ComparisonResult) => {
        this.comparisonResult.set(data);
        this.isLoading.set(false);
      })
      .catch((error) => {
        console.error('Comparison error:', error);
        this.errorMessage.set(`Error: ${error.message || 'Failed to compare images'}`);
        this.isLoading.set(false);
      });
  }

  onSliderChange(event: Event): void {
    const value = (event.target as HTMLInputElement).value;
    this.sliderValue.set(Number(value));
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

  getBandColor(): string {
    const band = this.comparisonResult()?.band;
    switch (band) {
      case 'Exact Duplicate':
        return '#ef4444';
      case 'Likely Duplicate':
        return '#f97316';
      case 'Similar – Same Family':
        return '#22c55e';
      case 'Similar – Related':
        return '#3b82f6';
      case 'Different':
        return '#9ca3af';
      default:
        return '#9ca3af';
    }
  }

  getSimilarityPercentage(): string {
    const score = this.comparisonResult()?.hybrid_score;
    if (score !== undefined && score !== null) {
      return (score * 100).toFixed(1);
    }
    return '0';
  }

  onClose(): void {
    this.closeCompare.emit();
  }
}
