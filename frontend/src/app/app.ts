import { Component, inject, signal, computed, ViewChild, ElementRef, AfterViewChecked, OnInit } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { UpperCasePipe } from '@angular/common';

interface ChatMessage {
  id: number;
  text: string;
  sender: 'user' | 'assistant' | 'typing' | 'system';
  timestamp: Date;
}

interface ChatResponse {
  response: string;
}

interface HealthResponse {
  status: string;
  custom_model_found: boolean;
}

interface UserProfile {
  name: string;
  level: string;
  income: string;
  concern: string;
}

interface RoadmapProfile {
  title: string;
  concepts: string[];
}

type BackendStatus = 'checking' | 'ready' | 'untrained' | 'offline';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [UpperCasePipe],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewChecked, OnInit {
  private http = inject(HttpClient);
  private apiBaseUrl = 'http://localhost:8000';
  private nextMessageId = 1;

  @ViewChild('scrollMe') private myScrollContainer!: ElementRef;

  // Onboarding
  onboardingComplete = signal(false);
  onboardingStep = signal(0);
  userProfile = signal<UserProfile>({ name: '', level: '', income: '', concern: '' });

  // Roadmap
  showRoadmap = signal(false);
  roadmapProfile = signal<RoadmapProfile>({ title: '', concepts: [] });

  // Chat
  messages = signal<ChatMessage[]>([]);
  inputText = signal('');
  isLoading = signal(false);
  backendStatus = signal<BackendStatus>('checking');

  onboardingQuestions = [
    { key: 'name', question: '¿Cómo te llamas?', placeholder: 'Tu nombre...' },
    { key: 'level', question: '¿Cuál es tu nivel de conocimiento financiero?', options: ['Principiante', 'Intermedio', 'Avanzado'] },
    { key: 'income', question: '¿Tienes ingresos propios?', options: ['Sí, trabajo', 'Sí, becas o ayudas', 'No de momento'] },
    { key: 'concern', question: '¿Qué te preocupa más?', options: ['Empezar a ahorrar', 'Entender la inversión', 'Salir de deudas', 'No sé por dónde empezar'] },
  ];

  currentQuestion = computed(() => {
    const step = this.onboardingStep();
    return step < this.onboardingQuestions.length ? this.onboardingQuestions[step] : null;
  });

  backendStatusLabel = computed(() => {
    switch (this.backendStatus()) {
      case 'ready':
        return 'MODELO ACTIVO';
      case 'untrained':
        return 'MODELO BASE';
      case 'offline':
        return 'BACKEND OFFLINE';
      default:
        return 'COMPROBANDO';
    }
  });

  canSendMessage = computed(() => {
    return !this.isLoading() && this.inputText().trim().length > 0;
  });

  onboardingInput = signal('');

  ngOnInit() {
    this.checkBackend();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    try {
      const el = this.myScrollContainer?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    } catch {
      // El scroll no debe bloquear la interfaz si la vista aun no esta lista.
    }
  }

  checkBackend() {
    this.backendStatus.set('checking');

    this.http.get<HealthResponse>(`${this.apiBaseUrl}/health`).subscribe({
      next: (res) => {
        this.backendStatus.set(res.custom_model_found ? 'ready' : 'untrained');
      },
      error: () => {
        this.backendStatus.set('offline');
      }
    });
  }

  updateOnboardingInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.onboardingInput.set(input.value);
  }

  selectOption(key: string, value: string) {
    this.userProfile.update(p => ({ ...p, [key]: value }));

    const step = this.onboardingStep();
    if (step + 1 < this.onboardingQuestions.length) {
      this.onboardingStep.set(step + 1);
    } else {
      this.finishOnboarding();
    }
  }

  submitOnboardingStep() {
    const value = this.onboardingInput().trim();
    if (!value) return;

    const step = this.onboardingStep();
    const key = this.onboardingQuestions[step].key;

    this.userProfile.update(p => ({ ...p, [key]: value }));
    this.onboardingInput.set('');

    if (step + 1 < this.onboardingQuestions.length) {
      this.onboardingStep.set(step + 1);
    } else {
      this.finishOnboarding();
    }
  }

  private generateProfile(): RoadmapProfile {
    const profile = this.userProfile();
    const concern = profile.concern;

    if (concern === 'Entender la inversión') {
      return {
        title: 'INVERSOR CURIOSO',
        concepts: ['Qué es un ETF', 'Fondos indexados', 'Riesgo y diversificación']
      };
    }

    if (concern === 'Salir de deudas') {
      return {
        title: 'GESTIÓN DE DEUDA',
        concepts: ['Método bola de nieve', 'Interés del crédito', 'Presupuesto de emergencia']
      };
    }

    return {
      title: 'AHORRADOR PRINCIPIANTE',
      concepts: ['Interés compuesto', 'Regla del 50/30/20', 'Cuenta de ahorro vs cuenta corriente']
    };
  }

  private finishOnboarding() {
    this.roadmapProfile.set(this.generateProfile());
    this.showRoadmap.set(true);
  }

  startTerminal() {
    const profile = this.userProfile();
    const rp = this.roadmapProfile();
    const lines = [
      'Hola, ' + profile.name + '. Perfil registrado.',
      'Tipo: ' + rp.title + '.',
      'Ruta inicial:',
      '- ' + rp.concepts.join(String.fromCharCode(10) + '- '),
      '',
      'Estoy listo para ayudarte con tus finanzas personales.',
      'Pregunta con tus palabras sobre ahorro, deudas, inversión, presupuesto o seguridad.'
    ];
    const greeting = lines.join(String.fromCharCode(10));
    this.messages.set([this.createMessage(greeting, 'assistant')]);
    this.onboardingComplete.set(true);
  }

  updateInput(event: Event) {
    const input = event.target as HTMLInputElement;
    this.inputText.set(input.value);
  }

  sendMessage() {
    const query = this.inputText().trim();
    if (!query || this.isLoading()) return;

    this.addMessage(query, 'user');
    this.inputText.set('');
    this.isLoading.set(true);
    this.addMessage('...', 'typing');

    this.http.post<ChatResponse>(`${this.apiBaseUrl}/chat`, {
      message: this.buildPrompt(query)
    }).subscribe({
      next: (res) => {
        const response = this.cleanResponse(res.response);
        this.replaceTypingMessage(response, 'assistant');
        this.isLoading.set(false);
        this.checkBackend();
      },
      error: (error: HttpErrorResponse) => {
        this.replaceTypingMessage(this.getErrorMessage(error), 'system');
        this.isLoading.set(false);
        this.backendStatus.set('offline');
      }
    });
  }

  private buildPrompt(userQuestion: string): string {
    const profile = this.userProfile();
    return [
      'Eres The Ticker, un asistente educativo de finanzas personales para jóvenes.',
      'No des asesoramiento financiero profesional ni prometas resultados.',
      `Perfil del usuario: nivel ${profile.level}; ingresos: ${profile.income}; interés principal: ${profile.concern}.`,
      `Pregunta del usuario: ${userQuestion}`
    ].join('\n');
  }

  private cleanResponse(response: string | null | undefined): string {
    const text = (response ?? '').trim();
    return text || 'No he podido generar una respuesta clara. Prueba a reformular la pregunta con más contexto.';
  }

  private getErrorMessage(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return 'No se pudo conectar con el backend. Comprueba que el servidor de Python esté arrancado en http://localhost:8000.';
    }

    const detail = typeof error.error?.detail === 'string' ? error.error.detail : 'Error inesperado en el servidor.';
    return `El backend respondió con un error (${error.status}): ${detail}`;
  }

  private createMessage(text: string, sender: ChatMessage['sender']): ChatMessage {
    return {
      id: this.nextMessageId++,
      text,
      sender,
      timestamp: new Date()
    };
  }

  private addMessage(text: string, sender: ChatMessage['sender']) {
    this.messages.update(messages => [...messages, this.createMessage(text, sender)]);
  }

  private replaceTypingMessage(text: string, sender: ChatMessage['sender']) {
    this.messages.update(messages => [
      ...messages.filter(message => message.sender !== 'typing'),
      this.createMessage(text, sender)
    ]);
  }
}
