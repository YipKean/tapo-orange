#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include "DFRobotDFPlayerMini.h"
#include <HardwareSerial.h>
#include "env_build.h"

#ifndef WIFI_SSID_ENV
#define WIFI_SSID_ENV "YOUR_WIFI_SSID"
#endif

#ifndef WIFI_PASS_ENV
#define WIFI_PASS_ENV "YOUR_WIFI_PASSWORD"
#endif

#ifndef PC_PING_URL_ENV
#define PC_PING_URL_ENV "http://192.168.1.100:8000/ping"
#endif

#ifndef API_TOKEN_ENV
#define API_TOKEN_ENV ""
#endif

#ifndef ESP32_STATIC_IP_ENV
#define ESP32_STATIC_IP_ENV ""
#endif

#ifndef ESP32_GATEWAY_ENV
#define ESP32_GATEWAY_ENV ""
#endif

#ifndef ESP32_SUBNET_ENV
#define ESP32_SUBNET_ENV "255.255.255.0"
#endif

#ifndef ESP32_DNS_ENV
#define ESP32_DNS_ENV ""
#endif

// Values are loaded from .env at build time by scripts/load_env.py.
const char* WIFI_SSID = WIFI_SSID_ENV;
const char* WIFI_PASS = WIFI_PASS_ENV;

// Optional reverse ping endpoint on your PC.
const char* PC_PING_URL = PC_PING_URL_ENV;
const char* API_TOKEN = API_TOKEN_ENV; // Optional: set if your API uses bearer auth.
const char* ESP32_STATIC_IP = ESP32_STATIC_IP_ENV;
const char* ESP32_GATEWAY = ESP32_GATEWAY_ENV;
const char* ESP32_SUBNET = ESP32_SUBNET_ENV;
const char* ESP32_DNS = ESP32_DNS_ENV;
const bool HAS_SD_CARD = false;

HardwareSerial mySerial( 1 ); // UART1
DFRobotDFPlayerMini player;
bool dfReady = false;
WebServer server( 80 );

const int MAX_AP_CANDIDATES = 8;
uint8_t apBssids[MAX_AP_CANDIDATES][6];
int32_t apChannels[MAX_AP_CANDIDATES];
int apRssis[MAX_AP_CANDIDATES];
String apBssidTexts[MAX_AP_CANDIDATES];
int apCandidateCount = 0;

const char* wifiStatusToText( wl_status_t status ) {
	switch ( status ) {
		case WL_NO_SHIELD:
			return "NO_SHIELD";
		case WL_IDLE_STATUS:
			return "IDLE";
		case WL_NO_SSID_AVAIL:
			return "NO_SSID_AVAIL";
		case WL_SCAN_COMPLETED:
			return "SCAN_COMPLETED";
		case WL_CONNECTED:
			return "CONNECTED";
		case WL_CONNECT_FAILED:
			return "CONNECT_FAILED";
		case WL_CONNECTION_LOST:
			return "CONNECTION_LOST";
		case WL_DISCONNECTED:
			return "DISCONNECTED";
		default:
			return "UNKNOWN";
	}
}

const char* wifiAuthModeToText( wifi_auth_mode_t mode ) {
	switch ( mode ) {
		case WIFI_AUTH_OPEN:
			return "OPEN";
		case WIFI_AUTH_WEP:
			return "WEP";
		case WIFI_AUTH_WPA_PSK:
			return "WPA_PSK";
		case WIFI_AUTH_WPA2_PSK:
			return "WPA2_PSK";
		case WIFI_AUTH_WPA_WPA2_PSK:
			return "WPA_WPA2_PSK";
		case WIFI_AUTH_WPA2_ENTERPRISE:
			return "WPA2_ENTERPRISE";
		case WIFI_AUTH_WPA3_PSK:
			return "WPA3_PSK";
		case WIFI_AUTH_WPA2_WPA3_PSK:
			return "WPA2_WPA3_PSK";
		case WIFI_AUTH_WAPI_PSK:
			return "WAPI_PSK";
		default:
			return "UNKNOWN";
	}
}

const char* wifiDisconnectReasonToText( uint8_t reason ) {
	switch ( reason ) {
		case 2:
			return "AUTH_EXPIRE";
		case 4:
			return "ASSOC_EXPIRE";
		case 5:
			return "ASSOC_TOOMANY";
		case 6:
			return "NOT_AUTHED";
		case 7:
			return "NOT_ASSOCED";
		case 8:
			return "ASSOC_LEAVE";
		case 15:
			return "4WAY_HANDSHAKE_TIMEOUT";
		case 201:
			return "NO_AP_FOUND";
		case 202:
			return "AUTH_FAIL";
		case 203:
			return "ASSOC_FAIL";
		case 204:
			return "HANDSHAKE_TIMEOUT";
		default:
			return "UNKNOWN";
	}
}

void onWiFiEvent( WiFiEvent_t event, WiFiEventInfo_t info ) {
	if ( event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED ) {
		const uint8_t reason = info.wifi_sta_disconnected.reason;
		Serial.print( "[WIFI] Disconnected, reason=" );
		Serial.print( reason );
		Serial.print( " (" );
		Serial.print( wifiDisconnectReasonToText( reason ) );
		Serial.println( ")" );
	}
}

void sortApCandidatesByRssiDesc() {
	for ( int i = 0; i < apCandidateCount - 1; ++i ) {
		for ( int j = i + 1; j < apCandidateCount; ++j ) {
			if ( apRssis[j] > apRssis[i] ) {
				const int rssiTmp = apRssis[i];
				apRssis[i] = apRssis[j];
				apRssis[j] = rssiTmp;

				const int32_t chTmp = apChannels[i];
				apChannels[i] = apChannels[j];
				apChannels[j] = chTmp;

				String textTmp = apBssidTexts[i];
				apBssidTexts[i] = apBssidTexts[j];
				apBssidTexts[j] = textTmp;

				for ( int k = 0; k < 6; ++k ) {
					const uint8_t b = apBssids[i][k];
					apBssids[i][k] = apBssids[j][k];
					apBssids[j][k] = b;
				}
			}
		}
	}
}

void scanWifiTargets() {
	Serial.println( "Scanning nearby WiFi..." );
	apCandidateCount = 0;

	const int count = WiFi.scanNetworks( false, true );
	if ( count <= 0 ) {
		Serial.println( "No WiFi networks found." );
		return;
	}

	for ( int i = 0; i < count; ++i ) {
		const String ssid = WiFi.SSID( i );
		const int rssi = WiFi.RSSI( i );
		const wifi_auth_mode_t auth = WiFi.encryptionType( i );

		if ( ssid == WIFI_SSID ) {
			Serial.print( "Target SSID found. RSSI=" );
			Serial.print( rssi );
			Serial.print( " dBm, auth=" );
			Serial.print( wifiAuthModeToText( auth ) );
			Serial.print( ", channel=" );
			Serial.print( WiFi.channel( i ) );
			Serial.print( ", bssid=" );
			Serial.println( WiFi.BSSIDstr( i ) );

			if ( apCandidateCount < MAX_AP_CANDIDATES ) {
				const uint8_t* bssid = WiFi.BSSID( i );
				for ( int j = 0; j < 6; ++j ) {
					apBssids[apCandidateCount][j] = bssid[j];
				}
				apChannels[apCandidateCount] = WiFi.channel( i );
				apRssis[apCandidateCount] = rssi;
				apBssidTexts[apCandidateCount] = WiFi.BSSIDstr( i );
				++apCandidateCount;
			}
		}
	}

	if ( apCandidateCount == 0 ) {
		Serial.println( "Target SSID not found in scan results." );
	} else {
		sortApCandidatesByRssiDesc();
		Serial.print( "AP candidates found: " );
		Serial.println( apCandidateCount );
		for ( int i = 0; i < apCandidateCount; ++i ) {
			Serial.print( "  [" );
			Serial.print( i );
			Serial.print( "] RSSI=" );
			Serial.print( apRssis[i] );
			Serial.print( ", channel=" );
			Serial.print( apChannels[i] );
			Serial.print( ", bssid=" );
			Serial.println( apBssidTexts[i] );
		}
	}

	WiFi.scanDelete();
}

void configureStaticNetworkIfNeeded() {
	if ( strlen( ESP32_STATIC_IP ) == 0 ) {
		Serial.println( "IP mode: DHCP" );
		return;
	}

	IPAddress localIp;
	IPAddress gateway;
	IPAddress subnet;
	IPAddress dns;

	const bool ipOk = localIp.fromString( ESP32_STATIC_IP );
	const bool gwOk = gateway.fromString( ESP32_GATEWAY );
	const bool subnetOk = subnet.fromString( ESP32_SUBNET );
	bool dnsOk = true;

	if ( strlen( ESP32_DNS ) > 0 ) {
		dnsOk = dns.fromString( ESP32_DNS );
	}

	if ( !ipOk || !gwOk || !subnetOk || !dnsOk ) {
		Serial.println( "Static IP config invalid in .env. Falling back to DHCP." );
		return;
	}

	bool configOk = false;
	if ( strlen( ESP32_DNS ) > 0 ) {
		configOk = WiFi.config( localIp, gateway, subnet, dns );
	} else {
		configOk = WiFi.config( localIp, gateway, subnet );
	}

	if ( configOk ) {
		Serial.print( "IP mode: STATIC (" );
		Serial.print( ESP32_STATIC_IP );
		Serial.println( ")" );
	} else {
		Serial.println( "WiFi.config failed. Falling back to DHCP." );
	}
}

void connectWiFi() {
	WiFi.setAutoReconnect( true );
	WiFi.persistent( false );
	WiFi.mode( WIFI_STA );
	WiFi.disconnect( true );
	delay( 200 );
	configureStaticNetworkIfNeeded();

	Serial.print( "Using SSID: [" );
	Serial.print( WIFI_SSID );
	Serial.println( "]" );

	while ( WiFi.status() != WL_CONNECTED ) {
		scanWifiTargets();

		if ( apCandidateCount == 0 ) {
			Serial.println( "No target AP found. Retrying scan in 10s..." );
			delay( 10000 );
			continue;
		}

		for ( int i = 0; i < apCandidateCount; ++i ) {
			Serial.print( "Trying AP [" );
			Serial.print( i );
			Serial.print( "] bssid=" );
			Serial.print( apBssidTexts[i] );
			Serial.print( ", channel=" );
			Serial.println( apChannels[i] );

			WiFi.disconnect( true );
			delay( 300 );
			WiFi.begin( WIFI_SSID, WIFI_PASS, apChannels[i], apBssids[i], true );

			unsigned long attemptStartMs = millis();
			unsigned long lastStatusPrintMs = 0;
			while ( millis() - attemptStartMs < 10000 ) {
				if ( WiFi.status() == WL_CONNECTED ) {
					break;
				}
				delay( 400 );
				Serial.print( "." );

				if ( millis() - lastStatusPrintMs >= 2000 ) {
					lastStatusPrintMs = millis();
					const wl_status_t st = WiFi.status();
					Serial.print( " status=" );
					Serial.print( static_cast<int>( st ) );
					Serial.print( " (" );
					Serial.print( wifiStatusToText( st ) );
					Serial.print( ")" );
				}
			}

			if ( WiFi.status() == WL_CONNECTED ) {
				break;
			}

			Serial.println();
			Serial.println( "AP attempt timed out after 10s, trying next AP..." );
		}
	}

	Serial.println();
	Serial.print( "WiFi connected. IP: " );
	Serial.println( WiFi.localIP() );
}

void ensureWiFi() {
	if ( WiFi.status() == WL_CONNECTED ) {
		return;
	}
	Serial.println( "WiFi dropped. Reconnecting..." );
	connectWiFi();
}

void addAuthHeaderIfNeeded( HTTPClient& http ) {
	if ( strlen( API_TOKEN ) > 0 ) {
		http.addHeader( "Authorization", String( "Bearer " ) + API_TOKEN );
	}
}

void playTrackIfReady( uint8_t track ) {
	if ( !HAS_SD_CARD ) {
		return;
	}
	if ( dfReady ) {
		player.play( track );
	}
}

int pingPcFromEsp32( String& bodyOut ) {
	ensureWiFi();

	HTTPClient http;
	http.begin( PC_PING_URL );
	addAuthHeaderIfNeeded( http );

	const int code = http.GET();
	Serial.print( "[PC_PING] HTTP " );
	Serial.println( code );

	if ( code > 0 ) {
		bodyOut = http.getString();
		Serial.print( "[PC_PING] Body: " );
		Serial.println( bodyOut );
	} else {
		bodyOut = http.errorToString( code );
		Serial.print( "[PC_PING] Error: " );
		Serial.println( bodyOut );
	}

	http.end();
	return code;
}

void handleHealth() {
	const String resp = "{\"ok\":true,\"device\":\"esp32\",\"service\":\"tapo-alarm\"}";
	server.send( 200, "application/json", resp );
}

void handleAlert() {
	const String body = server.arg( "plain" );
	Serial.print( "[ALERT] Body: " );
	Serial.println( body );

	// Default: incoming alert request triggers track 1.
	playTrackIfReady( 1 );

	server.send( 200, "application/json", "{\"ok\":true,\"action\":\"alert_received\"}" );
}

void handlePingPc() {
	String body;
	const int code = pingPcFromEsp32( body );
	const bool ok = code > 0;

	String safeBody = body;
	safeBody.replace( "\\", "\\\\" );
	safeBody.replace( "\"", "\\\"" );

	const String resp =
		String( "{\"ok\":" ) + ( ok ? "true" : "false" ) +
		",\"http_code\":" + String( code ) +
		",\"message\":\"" + safeBody + "\"}";

	server.send( ok ? 200 : 502, "application/json", resp );
}

void handlePlay() {
	const int track = server.arg( "track" ).toInt();
	if ( track <= 0 || track > 255 ) {
		server.send( 400, "application/json", "{\"ok\":false,\"error\":\"track query required (1-255)\"}" );
		return;
	}

	playTrackIfReady( static_cast<uint8_t>( track ) );
	server.send( 200, "application/json", "{\"ok\":true,\"action\":\"play_sent\"}" );
}

void setupHttpServer() {
	server.on( "/health", HTTP_GET, handleHealth );
	server.on( "/api/alert", HTTP_POST, handleAlert );
	server.on( "/api/ping-pc", HTTP_GET, handlePingPc );
	server.on( "/api/play", HTTP_GET, handlePlay );

	server.onNotFound( []() {
		server.send( 404, "application/json", "{\"ok\":false,\"error\":\"not_found\"}" );
	} );

	server.begin();
	Serial.println( "HTTP server listening on port 80" );
	Serial.println( "Routes: GET /health, POST /api/alert, GET /api/ping-pc, GET /api/play?track=1" );
}

void setup() {
	Serial.begin( 115200 );
	delay( 300 );
	WiFi.onEvent( onWiFiEvent );

	connectWiFi();

	// RX=16, TX=17, DFPlayer default 9600.
	mySerial.begin( 9600, SERIAL_8N1, 16, 17 );
	Serial.println( "Starting DFPlayer..." );

	if ( !player.begin( mySerial ) ) {
		Serial.println( "DFPlayer NOT detected. API tests will still run." );
	} else {
		Serial.println( "DFPlayer OK" );
		dfReady = true;
		player.volume( 25 );
		if ( HAS_SD_CARD ) {
			playTrackIfReady( 1 );
		} else {
			Serial.println( "SD card not inserted yet. Audio playback is disabled." );
		}
	}

	setupHttpServer();
	Serial.println( "Commands: type 'p' to ping PC from ESP32." );
}

void loop() {
	server.handleClient();

	if ( Serial.available() ) {
		const char cmd = static_cast<char>( Serial.read() );
		if ( cmd == 'p' || cmd == 'P' ) {
			String body;
			pingPcFromEsp32( body );
		}
	}
}
