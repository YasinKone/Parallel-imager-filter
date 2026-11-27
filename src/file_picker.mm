#import <Cocoa/Cocoa.h>

#include <string>

std::string open_image_file_picker() {
    @autoreleasepool {
        NSOpenPanel* panel = [NSOpenPanel openPanel];

        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];

        [panel setAllowedFileTypes:@[@"png", @"jpg", @"jpeg"]];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] firstObject];

            if (url != nil) {
                return std::string([[url path] UTF8String]);
            }
        }
    }

    return "";
}