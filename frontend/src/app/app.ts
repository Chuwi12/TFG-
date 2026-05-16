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

type BackendStatus = 'checking' | 'ready' | 'guided' | 'offline';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [UpperCasePipe],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewChecked, OnInit {
  private http = inject(HttpClient);
  private apiBaseUrl = this.resolveApiBaseUrl();
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
      case 'guided':
        return 'MODO GUIADO';
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
        this.backendStatus.set(res.custom_model_found ? 'ready' : 'guided');
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

    if (this.backendStatus() !== 'ready') {
      setTimeout(() => {
        this.replaceTypingMessage(this.generateGuidedResponse(query), 'assistant');
        this.isLoading.set(false);
        this.checkBackend();
      }, 350);
      return;
    }

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

  private generateGuidedResponse(userQuestion: string): string {
    const question = this.normalizeText(userQuestion);
    const profile = this.userProfile();
    const intro = `Con tu perfil de ${profile.level || 'principiante'}, lo enfocaría así:`;

    if (this.includesAny(question, ['ahorro', 'ahorrar', 'emergencia', 'guardar'])) {
      return [
        intro,
        '1. Separa primero un importe pequeño y fijo al recibir ingresos, aunque sean 5 o 10 euros.',
        '2. Crea un fondo de emergencia antes de pensar en invertir: el objetivo inicial puede ser cubrir un mes de gastos básicos.',
        '3. Revisa cada semana en qué se va el dinero para detectar gastos repetidos que puedas reducir sin complicarte.'
      ].join('\n');
    }

    if (this.includesAny(question, ['deuda', 'deudas', 'prestamo', 'credito', 'tarjeta'])) {
      return [
        intro,
        '1. Ordena tus deudas por cantidad pendiente, interés y fecha de pago.',
        '2. Prioriza las que tengan mayor interés, porque son las que más crecen con el tiempo.',
        '3. Evita pedir nueva deuda para tapar pagos antiguos salvo que entiendas claramente el coste total.'
      ].join('\n');
    }

    if (this.includesAny(question, ['invertir', 'inversion', 'acciones', 'etf', 'fondo', 'indexado', 'bolsa'])) {
      return [
        intro,
        '1. Antes de invertir, asegúrate de tener controlados tus gastos y un colchón básico.',
        '2. Aprende la diferencia entre rentabilidad y riesgo: una inversión puede subir, bajar o tardar años en compensar.',
        '3. Empieza estudiando productos diversificados, como fondos indexados o ETF, sin meter dinero que necesites a corto plazo.'
      ].join('\n');
    }

    if (this.includesAny(question, ['presupuesto', 'gastos', 'ingresos', 'sueldo', 'beca'])) {
      return [
        intro,
        '1. Divide tus ingresos en tres bloques: gastos necesarios, ocio y ahorro.',
        '2. La regla 50/30/20 puede servir como referencia, pero debes adaptarla a tu situación real.',
        '3. Lo importante no es que el presupuesto sea perfecto, sino que puedas revisarlo y mantenerlo cada mes.'
      ].join('\n');
    }

    if (this.includesAny(question, ['seguridad', 'estafa', 'cripto', 'riesgo', 'fraude'])) {
      return [
        intro,
        '1. Desconfía de cualquier propuesta que prometa beneficios rápidos o garantizados.',
        '2. No compartas claves, códigos ni datos bancarios fuera de canales oficiales.',
        '3. Si no entiendes cómo se gana dinero con un producto, trátalo como una señal de riesgo y párate antes de actuar.'
      ].join('\n');
    }

    return [
      intro,
      'Puedo ayudarte con ahorro, deudas, inversión, presupuesto o seguridad financiera.',
      'Para darte una respuesta más útil, formula la pregunta con una situación concreta, por ejemplo: "cobro 300 euros al mes y quiero empezar a ahorrar".'
    ].join('\n');
  }

  private getErrorMessage(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return `No se pudo conectar con el backend. Comprueba que el servidor de Python esté arrancado en ${this.apiBaseUrl}.`;
    }

    const detail = typeof error.error?.detail === 'string' ? error.error.detail : 'Error inesperado en el servidor.';
    return `El backend respondió con un error (${error.status}): ${detail}`;
  }

  private resolveApiBaseUrl(): string {
    const host = globalThis.location?.hostname || 'localhost';
    return `http://${host}:8000`;
  }

  private normalizeText(text: string): string {
    return text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  }

  private includesAny(text: string, words: string[]): boolean {
    return words.some(word => text.includes(word));
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
