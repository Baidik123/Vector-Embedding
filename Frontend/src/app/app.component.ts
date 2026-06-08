import { Component, ElementRef, ViewChild, signal, OnDestroy } from '@angular/core';
import { PdfCompare, AlgorithmId } from './pdf-compare/pdf-compare';

type UploadSide = 'first' | 'second';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [PdfCompare],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {

  @ViewChild('firstInput') firstInput!: ElementRef<HTMLInputElement>;
  @ViewChild('secondInput') secondInput!: ElementRef<HTMLInputElement>;

  firstFile = signal<File | null>(null);
  secondFile = signal<File | null>(null);

  firstPreview = signal<string | null>(null);
  secondPreview = signal<string | null>(null);

  firstError = signal('');
  secondError = signal('');

  firstDragOver = signal(false);
  secondDragOver = signal(false);

  showCompare = signal(false);

  selectedAlgorithms = signal<AlgorithmId[]>(['dhash']);

  // Raw object URLs so they can be revoked and memory freed
  private firstObjectUrl: string | null = null;
  private secondObjectUrl: string | null = null;

  ngOnDestroy(): void {
    this.revokeUrl('first');
    this.revokeUrl('second');
  }

  private revokeUrl(side: UploadSide): void {
    if (side === 'first' && this.firstObjectUrl) {
      URL.revokeObjectURL(this.firstObjectUrl);
      this.firstObjectUrl = null;
    } else if (side === 'second' && this.secondObjectUrl) {
      URL.revokeObjectURL(this.secondObjectUrl);
      this.secondObjectUrl = null;
    }
  }

  toggleAlgorithm(id: AlgorithmId): void {
    const current = this.selectedAlgorithms();
    if (current.includes(id)) {
      this.selectedAlgorithms.set(current.filter((a) => a !== id));
    } else {
      this.selectedAlgorithms.set([...current, id]);
    }
  }

  isAlgorithmSelected(id: AlgorithmId): boolean {
    return this.selectedAlgorithms().includes(id);
  }

  clearFile(side: UploadSide): void {
    this.revokeUrl(side);
    if (side === 'first') {
      this.firstFile.set(null);
      this.firstPreview.set(null);
      this.firstError.set('');
    } else {
      this.secondFile.set(null);
      this.secondPreview.set(null);
      this.secondError.set('');
    }
  }

  triggerFilePicker(side: UploadSide): void {
    if (side === 'first') {
      this.firstInput.nativeElement.click();
    } else {
      this.secondInput.nativeElement.click();
    }
  }

  onFileSelected(event: Event, side: UploadSide): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    if (!file) return;
    this.assignFile(file, side);
    input.value = '';
  }

  onDragOver(event: DragEvent, side: UploadSide): void {
    event.preventDefault();
    if (side === 'first') {
      this.firstDragOver.set(true);
    } else {
      this.secondDragOver.set(true);
    }
  }

  onDragLeave(event: DragEvent, side: UploadSide): void {
    event.preventDefault();
    if (side === 'first') {
      this.firstDragOver.set(false);
    } else {
      this.secondDragOver.set(false);
    }
  }

  onDrop(event: DragEvent, side: UploadSide): void {
    event.preventDefault();
    if (side === 'first') {
      this.firstDragOver.set(false);
    } else {
      this.secondDragOver.set(false);
    }
    const file = event.dataTransfer?.files?.[0] ?? null;
    if (!file) return;
    this.assignFile(file, side);
  }

  private assignFile(file: File, side: UploadSide): void {
    const isPng =
      file.type === 'image/png' || file.name.toLowerCase().endsWith('.png');

    if (!isPng) {
      if (side === 'first') {
        this.firstError.set('Only PNG files are allowed.');
        this.firstFile.set(null);
        this.firstPreview.set(null);
      } else {
        this.secondError.set('Only PNG files are allowed.');
        this.secondFile.set(null);
        this.secondPreview.set(null);
      }
      return;
    }

    // Revoke previous URL before creating a new one to prevent memory leaks
    this.revokeUrl(side);
    const objectUrl = URL.createObjectURL(file);

    if (side === 'first') {
      this.firstObjectUrl = objectUrl;
      this.firstError.set('');
      this.firstFile.set(file);
      this.firstPreview.set(objectUrl);
    } else {
      this.secondObjectUrl = objectUrl;
      this.secondError.set('');
      this.secondFile.set(file);
      this.secondPreview.set(objectUrl);
    }
  }

  formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  onCompare(): void {
    const first = this.firstFile();
    const second = this.secondFile();
    if (first && second && this.selectedAlgorithms().length > 0) {
      this.showCompare.set(true);
    }
  }

  onCloseCompare(): void {
    this.showCompare.set(false);
  }
}
