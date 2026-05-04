import { Component, OnInit, inject, signal, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterOutlet } from '@angular/router';

interface ChatMessage {
  id: number;
  sender: 'bot' | 'user';
  text: string;
  type: 'text' | 'loading';
}

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, AfterViewChecked {
  protected readonly title = signal('frontend');
  messages = signal<ChatMessage[]>([]);
  userInput: string = '';
  isApiReady = signal<boolean>(true);
  
  @ViewChild('chatScroll') chatScroll!: ElementRef;
  private http = inject(HttpClient);

  ngOnInit() {
    this.messages.set([
      { id: 1, sender: 'bot', text: '¡Hola! Soy tu asistente IA. ¿En qué puedo ayudarte hoy?', type: 'text' }
    ]);
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  scrollToBottom(): void {
    try {
      if (this.chatScroll) {
        this.chatScroll.nativeElement.scrollTop = this.chatScroll.nativeElement.scrollHeight;
      }
    } catch(err) { }
  }

  formatText(text: string): string {
    return text.replace(/\n/g, '<br>');
  }

  handleTextSubmit() {
    if (!this.userInput.trim() || !this.isApiReady()) return;
    
    const userText = this.userInput;
    this.messages.update(msgs => [...msgs, { id: Date.now(), sender: 'user', text: userText, type: 'text' }]);
    this.userInput = '';
    
    // Add loading indicator
    const loadingId = Date.now() + 1;
    this.messages.update(msgs => [...msgs, { id: loadingId, sender: 'bot', text: '', type: 'loading' }]);
    this.isApiReady.set(false);

    // Call backend
    this.http.post<{response: string}>('http://localhost:8000/chat', { message: userText }).subscribe({
      next: (res) => {
        this.messages.update(msgs => {
          const newMsgs = msgs.filter(m => m.id !== loadingId);
          return [...newMsgs, { id: Date.now(), sender: 'bot', text: res.response, type: 'text' }];
        });
        this.isApiReady.set(true);
      },
      error: (err) => {
        this.messages.update(msgs => {
          const newMsgs = msgs.filter(m => m.id !== loadingId);
          return [...newMsgs, { id: Date.now(), sender: 'bot', text: 'Error de comunicación con el servidor.', type: 'text' }];
        });
        this.isApiReady.set(true);
      }
    });
  }
}
