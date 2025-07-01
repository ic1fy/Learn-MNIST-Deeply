import pygame
import os
from PIL import Image

pygame.init()
canvas_size = 280  # 画布尺寸（10倍放大，方便手写）
cell_size = 28     # 输出尺寸

screen = pygame.display.set_mode((canvas_size, canvas_size))
pygame.display.set_caption("手写数字采集器 - 按 0~9 保存，C 清空")

output_dir = "TransferLearning/dataset_hand/digits"
os.makedirs(output_dir, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

drawing = False
last_pos = None
pen_radius = 12

def draw_circle(pos):
    pygame.draw.circle(screen, (255, 255, 255), pos, pen_radius)

def save_image(digit):
    raw_str = pygame.image.tostring(screen, 'RGB')
    img = Image.frombytes('RGB', (canvas_size, canvas_size), raw_str)
    img = img.convert('L')               # 转灰度
    img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
    folder = os.path.join(output_dir, str(digit))
    idx = len(os.listdir(folder))
    img.save(os.path.join(folder, f"img_{idx:04d}.png"))
    print(f"已保存数字 {digit} 到 {folder}")

def clear_screen():
    screen.fill((0, 0, 0))

clear_screen()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # 鼠标按下开始绘图
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

        # 键盘按键事件
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                clear_screen()
            elif pygame.K_0 <= event.key <= pygame.K_9:
                digit = event.key - pygame.K_0
                save_image(digit)
                clear_screen()
                
    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        if last_pos:
            pygame.draw.line(screen, (255, 255, 255), last_pos, mouse_pos, pen_radius*2)
        last_pos = mouse_pos

    pygame.display.flip()

pygame.quit()
