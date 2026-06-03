import { Component, ElementRef, ViewChild, signal, OnDestroy, inject } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { PdfCompare } from './pdf-compare/pdf-compare';

type UploadSide = 'first' | 'second';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [PdfCompare],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {
  private sanitizer = inject(DomSanitizer);

  @ViewChild('firstInput') firstInput!: ElementRef<HTMLInputElement>;
  @ViewChild('secondInput') secondInput!: ElementRef<HTMLInputElement>;

  firstFile = signal<File | null>(null);
  secondFile = signal<File | null>(null);

  firstPreview = signal<SafeResourceUrl | null>(null);
  secondPreview = signal<SafeResourceUrl | null>(null);

  firstError = signal('');
  secondError = signal('');

  firstDragOver = signal(false);
  secondDragOver = signal(false);

  showCompare = signal(false);

  ngOnDestroy(): void {
    this.cleanupPreviews();
  }

  private cleanupPreviews(): void {
    this.firstPreview.set(null);
    this.secondPreview.set(null);
  }

  clearFile(side: UploadSide): void {
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
      const message = 'Only PNG files are allowed.';
      if (side === 'first') {
        this.firstError.set(message);
        this.firstFile.set(null);
        this.firstPreview.set(null);
      } else {
        this.secondError.set(message);
        this.secondFile.set(null);
        this.secondPreview.set(null);
      }
      return;
    }

    const url = URL.createObjectURL(file);
    const previewUrl = this.sanitizer.bypassSecurityTrustResourceUrl(url);

    if (side === 'first') {
      this.firstError.set('');
      this.firstFile.set(file);
      this.firstPreview.set(previewUrl);
    } else {
      this.secondError.set('');
      this.secondFile.set(file);
      this.secondPreview.set(previewUrl);
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
    if (first && second) {
      this.showCompare.set(true);
    }
  }

  onCloseCompare(): void {
    this.showCompare.set(false);
  }
}
